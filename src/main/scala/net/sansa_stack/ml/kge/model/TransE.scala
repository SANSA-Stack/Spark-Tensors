package ml.dmlc.mxnet.spark.io

import ml.dmlc.mxnet._
import ml.dmlc.mxnet.{Symbol => s}

import scala.io.Source
import scala.util.Random
import ml.dmlc.mxnet.optimizer.{AdaGrad, Adam, SGD}
//import ml.dmlc.mxnet.spark.io.{LabeledPointIter, UnLabeledPointIter}
import ml.dmlc.mxnet.spark.io.{EvalMetrics, L2Similarity, MaxMarginLoss}
import ml.dmlc.mxnet.spark.{MXNDArray, MXNet, MXNetModel, MXnet}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{SparseMatrix, Vectors}

/**
  * Created by nilesh on 01/06/2017.
  */
class TransE(numEntities: Int, numRelations: Int, latentFactors: Int, batchSize: Int) {
  class Model(opt: Optimizer, trainSymbol: Symbol, scoreSymbol: Symbol, dataset: Seq[NDArray], paramNames: Seq[String], ctx: Context, dataName: String, labels: Option[Seq[NDArray]] = None) {
    private val (argDict, gradDict, auxParams) = {
      import ml.dmlc.mxnet.Xavier

      val initializer = new Xavier(factorType = "in", magnitude = 2.34f)

      //    val (argShapes, outputShapes, auxShapes) = transeModel.inferShape(
      //      (for (paramName <- paramNames) yield paramName -> Shape(batchSize, 5))
      //        toMap)

      println("dataset.head.shape")
      println(dataset.head.shape)
      val (argShapes, outputShapes, auxShapes) = trainSymbol.inferShape(Map(
        dataName -> dataset.head.shape
      ))
//        ++ (if(labels.isDefined) Map("linearregressionoutput0_label" -> Shape(batchSize, 1)) else Map()))

      //    argShapes.foreach(x => println(x))

      val argNames = trainSymbol.listArguments()
      val argDict = argNames.zip(argShapes.map(NDArray.empty(_, ctx))).toMap
      val gradDict = argNames.zip(argShapes).filter {
        case (name, shape) =>
          !paramNames.contains(name)
      }.map(x => x._1 -> NDArray.empty(x._2, ctx)).toMap
      argDict.foreach {
        case (name, ndArray) =>
          if (!paramNames.contains(name)) {
            initializer.initWeight(name, ndArray)
          }
      }

      val auxNames = trainSymbol.listAuxiliaryStates()
      val auxParams = (auxNames zip auxShapes).map { case (name, shape) =>
        (name, NDArray.zeros(shape, ctx))
      }.toMap

      (argDict, gradDict, auxParams)
    }

    val executor: Executor = trainSymbol.bind(ctx, argDict, gradDict)

    private val paramsGrads = gradDict.toList.zipWithIndex.map { case ((name, grad), idx) =>
      (idx, name, grad, opt.createState(idx, argDict(name)))
    }

    private var iter = 0

    def trainBatch() = {
      argDict(dataName).set(dataset(iter))
      labels.foreach(x => argDict("linearregressionoutput0_label").set(x(iter).reshape(Shape(1000))))
      iter += 1

      if (iter >= dataset.length) iter = 0

      executor.forward(isTrain = true)
      executor.backward()

      paramsGrads.foreach {
        case (idx, name, grad, optimState) =>
          opt.update(idx, argDict(name), grad, optimState)
      }
    }

    def getFeedforwardModel: FeedForward = {
      val model = FeedForward.newBuilder(scoreSymbol)
        .setContext(ctx)
        .setNumEpoch(0)
        .setBeginEpoch(0)
        .setOptimizer(opt)
        .setArgParams(argDict)
        .setAuxParams(auxParams)
        .setup()
      model
    }
  }

  def getNet(): (Symbol, Symbol, Symbol, Seq[String]) = {
    // embedding weight vectors
    val entityWeight = s.Variable("entity_weight")
    val relationWeight = s.Variable("relation_weight")

    def entityEmbedding(data: Symbol) =
      s.Embedding()()(Map("data" -> data, "weight" -> entityWeight, "input_dim" -> numEntities, "output_dim" -> latentFactors))

    def relationEmbedding(data: Symbol) =
      s.Embedding()()(Map("data" -> data, "weight" -> relationWeight, "input_dim" -> numRelations, "output_dim" -> latentFactors))

    // inputs
    val input = s.Variable("data")
//    val blah = s.flatten()()(Map("data" ->s.Variable("blah")))
//    val blah2 = s.expand_dims()()(Map("data" -> blah, "axis" -> 0))
    val splitInput = s.split()()(Map("data" -> input, "axis" -> 1, "num_outputs" -> 8))
    var head = splitInput.get(0)
    var relation = splitInput.get(1)
    var tail = splitInput.get(2)
    var corruptHead = splitInput.get(3)
    var corruptTail = splitInput.get(4)

    var litHead = splitInput.get(5)
    var litRelation = splitInput.get(6)
    var litValue = splitInput.get(7)

//    println(splitInput.inferShape(Shape(1000,5))._1.mkString(","))

//    var head = s.slice()()(Map("data" -> splitInput, "axis" -> 0, "begin" -> 0, "end" -> 1))
//    var relation = s.slice()()(Map("data" -> splitInput, "axis" -> 0, "begin" -> 1, "end" -> 2))
//    var tail = s.slice()()(Map("data" -> splitInput, "axis" -> 0, "begin" -> 2, "end" -> 3))
//    var corruptHead = s.slice()()(Map("data" -> splitInput, "axis" -> 0, "begin" -> 3, "end" -> 4))
//    var corruptTail = s.slice()()(Map("data" -> splitInput, "axis" -> 0, "begin" -> 4, "end" -> 5))

    head = entityEmbedding(head)
    relation = relationEmbedding(relation)
    tail = entityEmbedding(tail)
    corruptHead = entityEmbedding(corruptHead)
    corruptTail = entityEmbedding(corruptTail)
    litHead = entityEmbedding(litHead)
    litRelation = relationEmbedding(litRelation)

    var pred = head * relation
    pred = s.sum_axis()()(Map("data" -> pred, "axis" -> Shape(2)))
    val regressionLoss = s.Flatten()()(Map("data" -> s.square()()(Map("data" -> (litValue - pred)))))

    def getScore(head: Symbol, relation: Symbol, tail: Symbol) = DotSimilarity(head * relation, tail)

    val posScore = getScore(head, relation, tail)
    val negScore = getScore(corruptHead, relation, corruptTail)

    println(posScore)
    println(negScore)

    val loss = MaxMarginLoss(0.5f)(posScore, negScore, regressionLoss)

    // Prediction model
    val score = {
      val input = s.Variable("data")
      //    val blah = s.flatten()()(Map("data" ->s.Variable("blah")))
      //    val blah2 = s.expand_dims()()(Map("data" -> blah, "axis" -> 0))
      val splitInput = s.split()()(Map("data" -> input, "axis" -> 1, "num_outputs" -> 5))
      var head = splitInput.get(0)
      var relation = splitInput.get(1)
      var tail = splitInput.get(2)
      var corruptHead = splitInput.get(3)
      var corruptTail = splitInput.get(4)


      head = entityEmbedding(head)
      relation = relationEmbedding(relation)
      tail = entityEmbedding(tail)
      corruptHead = entityEmbedding(corruptHead)
      corruptTail = entityEmbedding(corruptTail)

      val posScore = getScore(head, relation, tail)
      val negScore = getScore(corruptHead, relation, corruptTail)

//      MaxMarginLoss(1.0f)(posScore, negScore)
      posScore
    }

    // Literals model
    val literalLoss = {
      val input = s.Variable("literaldata")
      val label = s.Variable("linearregressionoutput0_label")
      val splitInput = s.split()()(Map("data" -> input, "axis" -> 1, "num_outputs" -> 2))
      var head = splitInput.get(0)
      var relation = splitInput.get(1)
      head = entityEmbedding(head)
      relation = relationEmbedding(relation)
      var pred = head * relation
      pred = s.sum_axis()()(Map("data" -> pred, "axis" -> Shape(2)))
//      pred = s.Flatten()()(Map("data" -> pred))
      val loss = s.LinearRegressionOutput()()(Map("data" ->  pred, "label" -> label))
      loss
    }


    (loss, score, literalLoss, Seq("data", "literaldata", "blah"))
  }

  def train() = {
    val ctx = Context.gpu()
    val ctx2 = Context.gpu()
    //  val numEntities = 40943
    val (transeModel, scoreModel, literalModel, paramNames) = getNet()

    //    println(argDict("entity_weight").shape)


//    val conf = new SparkConf().setAppName("MXNet").setMaster("local[4]")
//    val sc = new SparkContext(conf)
//
//    val mxnet = new MXnet()
//      .setBatchSize(batchSize)
//      .setContext(Context.gpu()) // or GPU if you like
//      .setDimension(Shape(5))
//      .setNetwork(transeModel) // e.g. MLP model
//      .setNumEpoch(10)
////      .setNumServer(1)
//      .setNumWorker(1)
//      .setLabelName("blah")
//      // These jars are required by the KVStores at runtime.
//      // They will be uploaded and distributed to each node automatically
//      .setExecutorJars("/data/nilesh/Spark-Tensors/target/sansa-kge-0.0.1-SNAPSHOT-allinone.jar")
//    val input = readDataBatched("train").map{
//      case x =>
//        LabeledPoint(0.0, Vectors.dense(x.map(_.toDouble)))
//    }.toSeq
//    val model = mxnet.fit(sc.parallelize(input))
//    model.save(sc, "model.bin")

//    val model = MXNetModel.load(sc, "model.bin")

//    val dataIter = new LabeledPointIter(input.toIterator, Shape(5), 1000)

        var inputStuff = readDataBatched("train").map{
          case batch =>
            NDArray.array(batch.flatten.toArray, Shape(batch.size, batch.head.size))
        }

        val literals = Random.shuffle(readLiteralMatrix())
        var literalMatrix = NDArray.array(SparseMatrix.fromCOO(numEntities, numRelations, literals).toArray.map(_.toFloat),
          Shape(numEntities, numRelations))
        val means = NDArray.mean(literalMatrix, 0)
        val sd = NDArray.sqrt(NDArray.mean(NDArray.square(NDArray.broadcast_minus(literalMatrix, means)), 0))
        literalMatrix = NDArray.broadcast_div(NDArray.broadcast_minus(literalMatrix, means), sd)

        val literalData = {
          var literalCoordinates = literals.map{
            case (row: Int, col: Int, _) =>
              Seq(row.toFloat, col.toFloat)
          }

          literalCoordinates = literalCoordinates ++ literalCoordinates.take(batchSize - (literalCoordinates.size % batchSize))

          literalCoordinates.grouped(batchSize)
            .map(x => NDArray.array(x.flatten.toArray, Shape(batchSize, 2)))
            .toSeq
        }

        val literalLabels = {
          var literalValues = literals.map(_._3.toFloat)

          literalValues = literalValues ++ literalValues.take(batchSize - (literalValues.size % batchSize))

          literalValues.grouped(batchSize)
            .map(x => NDArray.array(x.toArray, Shape(batchSize, 1)))
            .toSeq
        }

    //        val (testSubjects, testRelations, testObjects, _, _) = readDataBatched("test")

        var literalAll = {
          var literalCoordValues = literals.map{
            case (row: Int, col: Int, value: Double) =>
              Seq(row, col, value.toFloat)
          }

          literalCoordValues = literalCoordValues ++ literalCoordValues.take(batchSize - (literalCoordValues.size % batchSize))

          literalCoordValues.grouped(batchSize)
            .map(x => NDArray.array(x.flatten.toArray, Shape(batchSize, 3)))
            .toSeq
        }


        if(inputStuff.length < literalAll.length) {
          inputStuff = inputStuff ++ inputStuff.take(literalAll.length - inputStuff.length)
        } else if(inputStuff.length > literalAll.length) {
          literalAll = literalAll ++ literalAll.take(inputStuff.length - literalAll.length)
        }

        val finalData = inputStuff.zip(literalAll).map {
          case (triples: NDArray, literals: NDArray) =>
            NDArray.concat(triples, literals, 2, 1).get
        }

        var minTestHits = 300f
        println(inputStuff.size)
        val opt = new AdaGrad()
        val trainer = new Model(opt, transeModel, scoreModel, finalData, paramNames, ctx, "data")
//        val literalTrainer = new Model(opt, literalModel, scoreModel, literalData, paramNames, ctx, "literaldata", labels = Some(literalLabels))
        for (iter <- 0 until inputStuff.size * 10) {
          trainer.trainBatch()
//          literalTrainer.trainBatch()
          //      println(s"iter $epoch, training Hits@1: ${Math.sqrt(Hits.hitsAt1(NDArray.ones(batchSize), executor.outputs(0)) / batchSize)}, min test Hits@1: $minTestHits")
          println(s"Entity model: iter $iter, training loss: ${trainer.executor.outputs(0).toArray.sum}")
//          println(s"Literal model: iter $iter, training loss: ${literalTrainer.executor.outputs(0).toArray.sum}")
        }


    val model2 = trainer.getFeedforwardModel

    val vectors = Random.shuffle(readDataBatched("train", test = false).flatten.map{
      case x =>
        LabeledPoint(0.0, Vectors.dense(x.map(_.toDouble)))
    }).take(50000).map(_.features)
    val dt = new MyPointIter(vectors.toIterator, Shape(5), batchSize, "data", "label", ctx)
//    println(dt.next().data.head.shape)
//    println(dt.next().data.size)
//    val predictions = model2.predict(dt).map(_.copyTo(ctx2))
//    println(model2.getArgParams.values.map(_.shape))
//    println(model2.getAuxParams.values.map(_.shape))
//      results.map(arr => MXNDArray(arr))
//      val predictions = model.predict(input.map(_.features).toIterator)

    val hits = new EvalMetrics(model2, 1, batchSize,
      NDArray.array(Array.range(0,numRelations).map(_.toFloat), Shape(1,numRelations), ctx2),
      new MyPointIter(vectors.toIterator, Shape(5), batchSize, "data", "label", ctx2), ctx2)
    println(hits.hits(5,10).mkString(","))
    println(hits.mrr)

    val vectors2 = readDataBatched("test", test = false).flatten.map{
      case x =>
        LabeledPoint(0.0, Vectors.dense(x.map(_.toDouble)))
    }.map(_.features)

    val hits2 = new EvalMetrics(model2, 1, batchSize,
      NDArray.array(Array.range(0,numRelations).map(_.toFloat), Shape(1,numRelations), ctx2),
      new MyPointIter(vectors2.toIterator, Shape(5), batchSize, "data", "label", ctx2), ctx2)
    println(hits2.hits(5,10).mkString(","))
    println(hits2.mrr)
//
//    println(predictions.mkString(","))
//
//    println(predictions(0).toArray.mkString(","))

    //    model.fit(trainData = dataIter,
//      evalData = null,
//      kvStore = kv)


////      if (epoch != 0 && epoch % 50 == 0) {
////        val tmp = for (i <- 0 until testSubjects.length) yield {
////          head.set(testSubjects(iter))
////          relation.set(testRelations(iter))
////          tail.set(testObjects(iter))
////
////          executor.forward(isTrain = false)
////          Hits.hitsAt1(NDArray.ones(batchSize), executor.outputs(0))
////        }
////        val testHits = Math.sqrt(tmp.toArray.sum / (testSubjects.length * batchSize))
////        if (testHits < minTestHits) minTestHits = testHits.toFloat
//      }
  }

  def hits(model: Symbol, at: Int, dataArray: NDArray) = {
    val input = readDataBatched("test")
  }

  def readDataBatched(stage: String, test: Boolean = false) = {
    val triplesFile = s"/data/yago/subgraph/exp/$stage.txt"
    val entityIDFile = "/data/yago/subgraph/exp/entity2id.txt"
    val relationIDFile = "/data/yago/subgraph/exp/relation2id.txt"

    def getIDMap(path: String) = Source.fromFile(path)
      .getLines()
      .map(_.split("\t"))
      .map(x => x(0) -> x(1).toFloat).toMap

    val entityID = getIDMap(entityIDFile)
    val relationID = getIDMap(relationIDFile)

    val triples = Source.fromFile(triplesFile).getLines().map(_.split("\t")).toSeq
    val mappedTriples = triples.map{
      case x =>
        Array(entityID(x(0)),
          relationID(x(1)),
          entityID(x(2))) ++
          (if(!test) Array(Random.nextInt(numEntities).toFloat,
        Random.nextInt(numEntities).toFloat)
        else Nil)
    }


    (mappedTriples ++ mappedTriples.take(batchSize - (mappedTriples.size % batchSize)))
      .grouped(batchSize).toSeq

    //        .map{
    //        case x =>
    //          x.flatMap(x => x).toArray
    //      }

    //        .map{
    //        case x =>
    //          NDArray.array((0 until 5).map(i => x.map(y => y(i))).reduce((x,y) => x ++ y).toArray, Shape(5, batchSize))
    //      }


    //      (triples.map(x => entityID(x(0))).toArray.grouped(batchSize).toSeq,
    //        triples.map(x => relationID(x(2))).toArray.grouped(batchSize).toSeq,
    //        triples.map(x => entityID(x(1))).toArray.grouped(batchSize).toSeq,
    //        triples.map(x => Random.nextInt(numEntities).toFloat).toArray.grouped(batchSize).toSeq,
    //        triples.map(x => Random.nextInt(numEntities).toFloat).toArray.grouped(batchSize).toSeq)
  }

  def readLiteralMatrix() = {
    val literalsFile = "/data/yago/subgraph/exp/literals.txt"
    val entityIDFile = "/data/yago/subgraph/exp/entity2id.txt"
    val relationIDFile = "/data/yago/subgraph/exp/relationwl2id.txt"

    def getIDMap(path: String) = Source.fromFile(path)
      .getLines()
      .map(_.split("\t"))
      .map(x => x(0) -> x(1).toInt).toMap

    val entityID = getIDMap(entityIDFile)
    val relationID = getIDMap(relationIDFile)

    val triples = Source.fromFile(literalsFile).getLines().map(_.split("\t")).toSeq
    val mappedTriples = triples.flatMap{
      case x =>
        try {
          Seq((entityID(x(0)),
            relationID(x(1)),
            x(3).toDouble))
        } catch {
          case ex: Exception =>
            Nil
        }
    }

    mappedTriples
  }
}