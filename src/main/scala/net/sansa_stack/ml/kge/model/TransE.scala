package ml.dmlc.mxnet.spark.io

import ml.dmlc.mxnet._
import ml.dmlc.mxnet.{Symbol => s}

import scala.io.Source
import scala.util.Random
import ml.dmlc.mxnet.optimizer.{Adam, SGD}
import ml.dmlc.mxnet.spark.io.{LabeledPointIter, UnLabeledPointIter}
import  ml.dmlc.mxnet.spark.io.{Hits, L2Similarity, MaxMarginLoss}
import ml.dmlc.mxnet.spark.{MXNDArray, MXNet, MXNetModel, MXnet}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors

/**
  * Created by nilesh on 01/06/2017.
  */
class TransE(numEntities: Int, numRelations: Int, latentFactors: Int, batchSize: Int) {
  def getNet(): (Symbol, Symbol, Seq[String]) = {
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
    val splitInput = s.split()()(Map("data" -> input, "axis" -> 1, "num_outputs" -> 5))
    var head = splitInput.get(0)
    var relation = splitInput.get(1)
    var tail = splitInput.get(2)
    var corruptHead = splitInput.get(3)
    var corruptTail = splitInput.get(4)

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

    def getScore(head: Symbol, relation: Symbol, tail: Symbol) = L2Similarity(head + relation, tail)

    val posScore = getScore(head, relation, tail)
    val negScore = getScore(corruptHead, relation, corruptTail)

    println(posScore)
    println(negScore)

    val loss = MaxMarginLoss(1.0f)(posScore, negScore)
    val (argShapes, outShapes, auxShapes) = loss.inferShape(Map("data" -> Shape(1000, 5)))
//    println(argShapes.mkString(","))
    println(outShapes.mkString(","))
    println(auxShapes.mkString(","))

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
      negScore
    }


    (loss, score, Seq("data", "blah"))
  }

  def train() = {
    val ctx = Context.cpu()
    //  val numEntities = 40943
    val (transeModel, scoreModel, paramNames) = getNet()

    import ml.dmlc.mxnet.Xavier

    val initializer = new Xavier(factorType = "in", magnitude = 2.34f)

//    val (argShapes, outputShapes, auxShapes) = transeModel.inferShape(
//      (for (paramName <- paramNames) yield paramName -> Shape(batchSize, 5))
//        toMap)

    val (argShapes, outputShapes, auxShapes) = transeModel.inferShape(Map(
      "data" -> Shape(batchSize, 5)
    ))

//    argShapes.foreach(x => println(x))

    val argNames = transeModel.listArguments()
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

    val auxNames = transeModel.listAuxiliaryStates()
    val auxParams = (auxNames zip auxShapes).map { case (name, shape) =>
      (name, NDArray.zeros(shape))
    }.toMap


    //    println(argDict("entity_weight").shape)


//    val conf = new SparkConf().setAppName("MXNet").setMaster("local[4]")
//    val sc = new SparkContext(conf)
//
//    val mxnet = new MXnet()
//      .setBatchSize(batchSize)
//      .setContext(Context.cpu()) // or GPU if you like
//      .setDimension(Shape(5))
//      .setNetwork(transeModel) // e.g. MLP model
//      .setNumEpoch(10)
////      .setNumServer(1)
//      .setNumWorker(1)
//      .setLabelName("blah")
//      // These jars are required by the KVStores at runtime.
//      // They will be uploaded and distributed to each node automatically
//      .setExecutorJars("/home/nilesh/utils/Spark-Tensors/target/sansa-kge-0.0.1-SNAPSHOT-allinone.jar")
//    val input = readDataBatched("train").map{
//      case x =>
//        LabeledPoint(0.0, Vectors.dense(x.map(_.toDouble)))
//    }.toSeq
//    val model = mxnet.fit(sc.parallelize(input))
//    model.save(sc, "model.bin")

//    val model = MXNetModel.load(sc, "model.bin")

//    val dataIter = new LabeledPointIter(input.toIterator, Shape(5), 1000)

        val executor = transeModel.bind(ctx, argDict, gradDict)

        val opt = new Adam(learningRate = 0.01f, wd = 0.0001f)
        val paramsGrads = gradDict.toList.zipWithIndex.map { case ((name, grad), idx) =>
          (idx, name, grad, opt.createState(idx, argDict(name)))
        }

        val input = argDict("data")
//        val blah = argDict("blah")

        val inputStuff = readDataBatched("train")

    //    val (testSubjects, testRelations, testObjects, _, _) = readDataBatched("test")

        var iter = 0
        var minTestHits = 100f
        for (epoch <- 0 until 5) {
          input.set(inputStuff(iter).flatten.toArray)
//          blah.set(NDArray.zeros(batchSize, 1))
          iter += 1

          if (iter >= inputStuff.length) iter = 0

          executor.forward(isTrain = true)
          executor.backward()

          paramsGrads.foreach {
            case (idx, name, grad, optimState) =>
              opt.update(idx, argDict(name), grad, optimState)
          }

          //      println(s"iter $epoch, training Hits@1: ${Math.sqrt(Hits.hitsAt1(NDArray.ones(batchSize), executor.outputs(0)) / batchSize)}, min test Hits@1: $minTestHits")

          println(s"iter $epoch, training loss: ${executor.outputs(0).toArray.sum}")
        }

    val optimizer: Optimizer = new SGD(learningRate = 0.01f,
      momentum = 0.9f, wd = 0.00001f)
    val model2 = FeedForward.newBuilder(scoreModel)
      .setContext(Context.cpu())
      .setNumEpoch(0)
      .setBeginEpoch(0)
      .setOptimizer(optimizer)
      .setArgParams(argDict
//        .map{
//        case params @ (str: String, array: NDArray) =>
//          if(str.contains("data"))
//            (str, NDArray.slice_axis(array, 1, 0, 3).get)
//          else
//            params
//      }
      )
      .setAuxParams(auxParams)
      .setup()
//


    val vectors = readDataBatched("train", test = false).flatten.map{
      case x =>
        LabeledPoint(0.0, Vectors.dense(x.map(_.toDouble)))
    }.take(1000).map(_.features).toIterator
    println(vectors.next().size)
    val dt = new PointIter(vectors, Shape(5), 1000, "data")
//    println(dt.next().data.head.shape)
//    println(dt.next().data.size)
//    val predictions = model2.predict(dt)
//    println(model2.getArgParams.values.map(_.shape))
//    println(model2.getAuxParams.values.map(_.shape))
//      results.map(arr => MXNDArray(arr))
//      val predictions = model.predict(input.map(_.features).toIterator)

    val hits = new Hits(model2, 1, NDArray.array(Array.range(0,18).map(_.toFloat),
      Shape(1,18)), dt)
    println(hits.score(10))

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
    val triplesFile = s"/home/nilesh/utils/Spark-Tensors/data/$stage.txt"
    val entityIDFile = "/home/nilesh/utils/Spark-Tensors/data/entity2id.txt"
    val relationIDFile = "/home/nilesh/utils/Spark-Tensors/data/relation2id.txt"


    def getIDMap(path: String) = Source.fromFile(path)
      .getLines()
      .map(_.split("\t"))
      .map(x => x(0) -> x(1).toFloat).toMap

    val entityID = getIDMap(entityIDFile)
    val relationID = getIDMap(relationIDFile)

    val triples = Random.shuffle(Source.fromFile(triplesFile).getLines().map(_.split("\t")).toSeq)
    triples.map{
      case x =>
        Array(entityID(x(0)),
          relationID(x(2)),
          entityID(x(1))) ++
          (if(!test) Array(Random.nextInt(numEntities).toFloat,
        Random.nextInt(numEntities).toFloat)
        else Nil)
    }
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
}