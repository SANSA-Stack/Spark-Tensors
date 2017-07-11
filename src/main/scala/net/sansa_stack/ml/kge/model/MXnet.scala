package ml.dmlc.mxnet.spark

import ml.dmlc.mxnet.optimizer.{Adam, SGD}
import ml.dmlc.mxnet.spark.io.{LabeledPointIter, UnLabeledPointIter}
import ml.dmlc.mxnet._
import ml.dmlc.mxnet.spark._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.slf4j.{Logger, LoggerFactory}

/**
  * Created by nilesh on 29/6/17.
  */
class MXnet extends Serializable {
  private val logger: Logger = LoggerFactory.getLogger(classOf[MXnet])
  private val params: MXNetParams = new MXNetParams

  def setBatchSize(batchSize: Int): this.type = {
    params.batchSize = batchSize
    this
  }

  def setNumEpoch(numEpoch: Int): this.type = {
    params.numEpoch = numEpoch
    this
  }

  def setDimension(dimension: Shape): this.type = {
    params.dimension = dimension
    this
  }

  def setNetwork(network: Symbol): this.type = {
    params.setNetwork(network)
    this
  }

  def setContext(ctx: Array[Context]): this.type = {
    params.context = ctx
    this
  }

  def setNumWorker(numWorker: Int): this.type = {
    params.numWorker = numWorker
    this
  }

  def setNumServer(numServer: Int): this.type = {
    params.numServer = numServer
    this
  }

  def setDataName(name: String): this.type = {
    params.dataName = name
    this
  }

  def setLabelName(name: String): this.type = {
    params.labelName = name
    this
  }

  /**
    * The application (including parameter scheduler & servers)
    * will exist if it hasn't received heart beat for over timeout seconds
    * @param timeout timeout in seconds (default 300)
    */
  def setTimeout(timeout: Int): this.type = {
    params.timeout = timeout
    this
  }

  /**
    * These jars are required by the KVStores at runtime.
    * They will be uploaded and distributed to each node automatically
    * @param jars jars required by the KVStore at runtime.
    */
  def setExecutorJars(jars: String): this.type = {
    params.jars = jars.split(",|:")
    this
  }

  def setJava(java: String): this.type = {
    params.javabin = java
    this
  }

  def fit(data: RDD[LabeledPoint]) = {
    val sc = data.context
    // distribute native jars
    params.jars.foreach(jar => sc.addFile(jar))

    val trainData = {
      if (params.numWorker > data.partitions.length) {
        logger.info("repartitioning training set to {} partitions", params.numWorker)
        data.repartition(params.numWorker)
      } else if (params.numWorker < data.partitions.length) {
        logger.info("repartitioning training set to {} partitions", params.numWorker)
        data.coalesce(params.numWorker)
      } else {
        data
      }
    }

    val schedulerIP = utils.Network.ipAddress
    val schedulerPort = utils.Network.availablePort
    // TODO: check ip & port available
    logger.info("Starting scheduler on {}:{}", schedulerIP, schedulerPort)
    val scheduler = new ParameterServer(params.runtimeClasspath, role = "scheduler",
      rootUri = schedulerIP, rootPort = schedulerPort,
      numServer = params.numServer, numWorker = params.numWorker,
      timeout = params.timeout, java = params.javabin)
    require(scheduler.startProcess(), "Failed to start ps scheduler process")

    sc.parallelize(1 to params.numServer, params.numServer).foreachPartition { p =>
      logger.info("Starting server ...")
      val server = new ParameterServer(params.runtimeClasspath,
        role = "server",
        rootUri = schedulerIP, rootPort = schedulerPort,
        numServer = params.numServer,
        numWorker = params.numWorker,
        timeout = params.timeout,
        java = params.javabin)
      require(server.startProcess(), "Failed to start ps server process")
    }

    val job = trainData.mapPartitions { partition =>
      val dataIter = new UnLabeledPointIter(
        partition, params.dimension,
        params.batchSize,
        dataName = params.dataName)

      // TODO: more nature way to get the # of examples?
      var numExamples = 0
      while (dataIter.hasNext) {
        val dataBatch = dataIter.next()
        numExamples += dataBatch.label.head.shape(0)
      }
      logger.debug("Number of samples: {}", numExamples)
      dataIter.reset()

      logger.info("Launching worker ...")
      logger.info("Batch {}", params.batchSize)
      // give enough time for ps-lite to detect the dead nodes
      Thread.sleep(2000)
      KVStoreServer.init(ParameterServer.buildEnv(role = "worker",
        rootUri = schedulerIP, rootPort = schedulerPort,
        numServer = params.numServer,
        numWorker = params.numWorker))
      val kv = KVStore.create("dist_async")
      kv.setBarrierBeforeExit(false)

      val optimizer: Optimizer = new SGD(learningRate = 0.01f,
        momentum = 0.9f, wd = 0.00001f)

      logger.debug("Define model")
      val model = new FeedForward(ctx = params.context,
        symbol = params.getNetwork,
        numEpoch = params.numEpoch,
        optimizer = optimizer,
        initializer = new Xavier(factorType = "in", magnitude = 2.34f),
        argParams = null,
        auxParams = null,
        beginEpoch = 0,
        epochSize = numExamples / params.batchSize / kv.numWorkers)
      logger.info("Start training ...")
      model.fit(trainData = dataIter,
        evalData = null,
        kvStore = kv)

      logger.info("Training finished, waiting for other workers ...")
      dataIter.dispose()
      println("Anything after this? 1")
      kv.setBarrierBeforeExit(true)
      println("Anything after this? 2")
      kv.dispose()
      println("Anything after this? 3")
      Iterator(new MXNetModel(
        model, params.dimension, params.batchSize,
        dataName = params.dataName, labelName = params.labelName))
    }.cache()

    // force job to run
    job.foreachPartition(_ => logger.info("This partition did fuck-all!"))
    println("Anything after this? 4")
    // simply the first model
    val mxModel = job.first()

    logger.info("Waiting for scheduler ...")
    scheduler.waitFor()
    mxModel
  }
//
//  private def initSymbolParams(trainData: DataIter)
//  : (IndexedSeq[String], IndexedSeq[String], IndexedSeq[String]) = {
//    if (symGen != null) {
//      this.symbol = symGen.generate(trainData.defaultBucketKey)
//      checkArguments()
//    }
//    initParams(trainData.provideData ++ trainData.provideLabel)
//  }
//
//  def checkArguments(): Unit = {
//    // check if symbol contain duplicated names.
//    ExecutorManager.checkArguments(symbol)
//    // rematch parameters to delete useless ones
//    if (allowExtraParams) {
//      if (_argParams != null) {
//        val argNames = symbol.listArguments().toSet
//        _argParams = _argParams.filter { case (k, v) => argNames.contains(k) }
//      }
//      if (auxParams != null) {
//        val auxNames = symbol.listAuxiliaryStates().toSet
//        _auxParams = _auxParams.filter { case (k, v) => auxNames.contains(k) }
//      }
//    }
//  }
//
//  private def initParams(symbol: Symbol, inputShapes: Map[String, Shape], overwrite: Boolean = false)
//  : (IndexedSeq[String], IndexedSeq[String], IndexedSeq[String]) = {
//    val (argShapes, _, auxShapes) = symbol.inferShape(inputShapes)
//    val argNames = symbol.listArguments()
//    val inputNames = inputShapes.keys.toSet
//    val paramNames = argNames.filter(!inputNames.contains(_))
//    val auxNames = symbol.listAuxiliaryStates()
//
//    val paramNameShapes = (argNames zip argShapes).filter { case (name, _) =>
//      paramNames.contains(name)
//    }
//    val argParams = paramNameShapes.map { case (name, shape) =>
//      (name, NDArray.zeros(shape))
//    }.toMap
//    val auxParams = (auxNames zip auxShapes).map { case (name, shape) =>
//      (name, NDArray.zeros(shape))
//    }.toMap
//
//    for ((k, v) <- argParams) {
//      if (_argParams != null && _argParams.contains(k) && (!overwrite)) {
//        argParams(k).set(_argParams(k))
//      } else {
//        initializer(k, v)
//      }
//    }
//
//    for ((k, v) <- auxParams) {
//      if (_auxParams != null && _auxParams.contains(k) && (!overwrite)) {
//        auxParams(k).set(_auxParams(k))
//      } else {
//        initializer(k, v)
//      }
//    }
//
//    _argParams = argParams
//    _auxParams = auxParams
//    (argNames, paramNames, auxNames)
//  }

}
