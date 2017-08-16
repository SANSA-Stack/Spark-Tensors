package ml.dmlc.mxnet.spark.io

import breeze.linalg.DenseMatrix
import ml.dmlc.mxnet._
import ml.dmlc.mxnet.module.Module
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
  class Model(opt: Optimizer,
              trainSymbol: Symbol,
              scoreSymbol: Symbol,
              dataset: Seq[NDArray],
              literalDataset: Seq[NDArray],
              paramNames: Seq[String],
              ctx: Context,
              dataName: String,
              labels: Seq[NDArray],
             literalMatrix: NDArray) {
    private val (argDict, gradDict, auxParams) = {
      import ml.dmlc.mxnet.Xavier

      val initializer = new Uniform()

      //    val (argShapes, outputShapes, auxShapes) = transeModel.inferShape(
      //      (for (paramName <- paramNames) yield paramName -> Shape(batchSize, 5))
      //        toMap)

      println("dataset.head.shape")
      println(dataset.head.shape)

      println(trainSymbol.listArguments().mkString("|"))
      println(scoreSymbol.listArguments().mkString("|"))
      val (a, b, c) = trainSymbol.inferShape(Map(
        dataName -> Shape(batchSize,5),
        "literal_matrix" -> Shape(numEntities, numRelations)
//        "literaldata" -> Shape(1000,2)
//        "linearregressionoutput0_label" -> Shape(batchSize)
      ))

      val (argShapes, outputShapes, auxShapes) = (a,b,c)

//        ++ (if(labels.isDefined) Map("linearregressionoutput0_label" -> Shape(batchSize, 1)) else Map()))

      //    argShapes.foreach(x => println(x))

      val argNames = (trainSymbol.listArguments()).distinct
      for(i <- argNames){
        println(i)
      }
      for(i <- argShapes){
        println(i)
      }
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
          if(paramNames.contains("literal_matrix")) {
            ndArray.set(literalMatrix)
          }
      }

      val auxNames = (trainSymbol.listAuxiliaryStates()).distinct
      val auxParams = (auxNames zip auxShapes).map { case (name, shape) =>
        (name, NDArray.zeros(shape, ctx))
      }.toMap

      (argDict, gradDict, auxParams)
    }

    val tripleExecutor: Executor = trainSymbol.bind(ctx, argDict, gradDict)
//    val scoreExecutor: Executor = scoreSymbol.bind(ctx, argDict, gradDict)
//      Map("data" -> "null",
//      "literaldata" -> "null",
//      "entity_weight" -> "write",
//      "relation_weight" -> "write",
//      "literal_entity_weight" -> "write",
//      "literal_relation_weight" -> "write"), Nil, null, null)

//    trainSymbol.bind
//    val literalExecutor: Executor = literalSymbol.bind(ctx, argDict, gradDict)

//    val module = new Module(trainSymbol, Array("data"), Array("linearregressionoutput0_label"),
//      contexts = Context.gpu())
//    module.bind(Array(DataDesc("data", Shape(1000,7))), Some(Array(DataDesc("linearregressionoutput0_label", Shape(1000))))
//
    private val paramsGrads = gradDict.toList.zipWithIndex.map { case ((name, grad), idx) =>
      (idx, name, grad, opt.createState(idx, argDict(name)))
    }

    private var tripleIter = 0
    private var literalIter = 0

    def trainTripleBatch() = {
      argDict(dataName).set(dataset(tripleIter))
      tripleIter += 1

      if (tripleIter >= dataset.length) tripleIter = 0

      tripleExecutor.forward(isTrain = true)
      tripleExecutor.backward()
//      scoreExecutor.forward(isTrain = false)

      paramsGrads.foreach {
        case (idx, name, grad, optimState) =>
//          val older = argDict(name).copy()
//          if(!name.contains("literal_matrix"))
          opt.update(idx, argDict(name), grad, optimState)
//          val updated = argDict(name)

//          println(name + " " + NDArray.sum(NDArray.square(updated-older)).toScalar)
//          older.dispose()
      }
    }

//    def trainLiteralBatch() = {
//      argDict("literaldata").set(literalDataset(literalIter))
//      argDict("linearregressionoutput0_label").set(labels(literalIter).reshape(Shape(batchSize)))
//      literalIter += 1
//
//      if (literalIter >= literalDataset.length) literalIter = 0
//
//      tripleExecutor.forward(isTrain = false)
////      tripleExecutor.backward(outGrad = tripleExecutor.outputs(1))
//
//      paramsGrads.foreach {
//        case (idx, name, grad, optimState) =>
//          opt2.update(idx, argDict(name), grad, optimState)
//      }
//    }

    def getFeedforwardModel: FeedForward = {
      val model = FeedForward.newBuilder(scoreSymbol)
        .setContext(ctx)
        .setNumEpoch(0)
        .setBeginEpoch(0)
        .setOptimizer(opt)
        .setArgParams(argDict)
        .setAuxParams(auxParams)
        .setAllowExtraParams(true)
        .setup()
      model
    }
  }

  def getNet() = {
    // embedding weight vectors
    val entityWeight = s.Variable("entity_weight")
    val literalEntityWeight = s.Variable("literal_entity_weight")
    val relationWeight = s.Variable("relation_weight")
    val literalRelationWeight = s.Variable("literal_relation_weight")
    val literalMatrix = s.Variable("literal_matrix")

    def entityEmbedding(data: Symbol) =
      s.Embedding()()(Map("data" -> data, "weight" -> entityWeight, "input_dim" -> numEntities, "output_dim" -> latentFactors))

//    def literalEntityEmbedding(data: Symbol) =
//      s.Embedding()()(Map("data" -> s.BlockGrad()()(Map("data" -> data)), "weight" -> literalEntityWeight, "input_dim" -> numEntities, "output_dim" -> latentFactors))

    def literalMatrixEmbedding(data: Symbol) =
      s.Embedding()()(Map("data" -> s.BlockGrad()()(Map("data" -> data)), "weight" -> literalMatrix, "input_dim" -> numEntities, "output_dim" -> numRelations))

    def relationEmbedding(data: Symbol) =
      s.Embedding()()(Map("data" -> data, "weight" -> relationWeight, "input_dim" -> numRelations, "output_dim" -> latentFactors))

//    def literalRelationEmbedding(data: Symbol) =
//      s.Embedding()()(Map("data" -> s.BlockGrad()()(Map("data" -> data)), "weight" -> literalRelationWeight, "input_dim" -> numRelations, "output_dim" -> latentFactors))

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

//    var litHead = splitInput.get(5)
//    var litRelation = splitInput.get(6)
//    var litValue = splitInput.get(7)

//    println(splitInput.inferShape(Shape(1000,5))._1.mkString(","))

//    var head = s.slice()()(Map("data" -> splitInput, "axis" -> 0, "begin" -> 0, "end" -> 1))
//    var relation = s.slice()()(Map("data" -> splitInput, "axis" -> 0, "begin" -> 1, "end" -> 2))
//    var tail = s.slice()()(Map("data" -> splitInput, "axis" -> 0, "begin" -> 2, "end" -> 3))
//    var corruptHead = s.slice()()(Map("data" -> splitInput, "axis" -> 0, "begin" -> 3, "end" -> 4))
//    var corruptTail = s.slice()()(Map("data" -> splitInput, "axis" -> 0, "begin" -> 4, "end" -> 5))

    val headLiterals = s.reshape()()(Map("data" -> literalMatrixEmbedding(head), "shape" -> Shape(batchSize, 1, numRelations)))
    val tailLiterals = s.reshape()()(Map("data" -> literalMatrixEmbedding(tail), "shape" -> Shape(batchSize, 1, numRelations)))
    val corruptHeadLiterals = s.reshape()()(Map("data" -> literalMatrixEmbedding(corruptHead), "shape" -> Shape(batchSize, 1, numRelations)))
    val corruptTailLiterals = s.reshape()()(Map("data" -> literalMatrixEmbedding(corruptTail), "shape" -> Shape(batchSize, 1, numRelations)))

    head = entityEmbedding(head)
    relation = relationEmbedding(relation)
    tail = entityEmbedding(tail)
    corruptHead = entityEmbedding(corruptHead)
    corruptTail = entityEmbedding(corruptTail)

    def densify(literalFeatures: Symbol) = {
      var litRelation = s.FullyConnected()()(Map("data" -> literalFeatures, "num_hidden" -> numRelations*2))
      litRelation = s.FullyConnected()()(Map("data" -> litRelation, "num_hidden" -> 30))
      litRelation = s.Dropout()()(Map("data" -> litRelation, "p" -> 0.5))
      litRelation = s.reshape()()(Map("data" -> litRelation, "shape" -> Shape(batchSize, 1, 30)))
      litRelation
    }

    def dropout(symbol: Symbol, p: Float = 0.5f) = s.Dropout()()(Map("data" -> symbol, "p" -> p))

//    head = s.concat()(head, densify(headLiterals))(Map("num_args" -> 2,  "dim" -> 2))
////    relation = s.concat()(relation, s.ones_like()()(Map("data" -> headLiterals)))(Map("num_args" -> 2,  "dim" -> 2))
//    tail = s.concat()(tail, densify(tailLiterals))(Map("num_args" -> 2,  "dim" -> 2))
//    corruptHead = s.concat()(corruptHead, densify(corruptHeadLiterals))(Map("num_args" -> 2,  "dim" -> 2))
//    corruptTail = s.concat()(corruptTail, densify(corruptTailLiterals))(Map("num_args" -> 2,  "dim" -> 2))
//
//    def getScore(head: Symbol, relation: Symbol, tail: Symbol) = {
////      DotSimilarity(head * relation, tail)
//        L2Similarity(head + relation, tail)
//    }

    def getScore(head: Symbol, headLiterals: Symbol, relation: Symbol, tail: Symbol, tailLiterals: Symbol) = {
      var spo = s.concat()(head, headLiterals, relation, tail, tailLiterals)(Map("num_args" -> 5,  "dim" -> 2))
//      spo = s.FullyConnected()()(Map("data" -> spo, "num_hidden" -> latentFactors*2))
//      spo = s.Dropout()()(Map("data" -> spo, "p" -> 0.5))
      spo = s.FullyConnected()()(Map("data" -> spo, "num_hidden" -> latentFactors))
      spo = s.Dropout()()(Map("data" -> spo, "p" -> 0.5))
      spo = s.Activation()()(Map("data" -> spo, "act_type" -> "tanh"))
      spo = s.FullyConnected()()(Map("data" -> spo, "num_hidden" -> 1))
      spo
    }

    val posScore = getScore(head, headLiterals, relation, tail, tailLiterals)
    val negScore = getScore(corruptHead, corruptHeadLiterals, relation, corruptTail, corruptTailLiterals)

    println(posScore)
    println(negScore)

    val loss = MaxMarginLoss(50.0f)(posScore, negScore)
//    val positiveOutput = s.BlockGrad()()(Map("data" -> posScore))
//    val negativeOutput = s.BlockGrad()()(Map("data" -> negScore))

    //    val difference = s.make_loss(name = "loss2")()(Map("data" -> s.BlockGrad()()(Map("data" -> s.Flatten()()(Map("data" ->(negScore - posScore)))))))

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

//      val headLiterals = literalMatrixEmbedding(head)
//      val tailLiterals = literalMatrixEmbedding(tail)
//      val corruptHeadLiterals = literalMatrixEmbedding(corruptHead)
//      val corruptTailLiterals = literalMatrixEmbedding(corruptTail)


      val headLiterals = s.reshape()()(Map("data" -> literalMatrixEmbedding(head), "shape" -> Shape(batchSize, 1, numRelations)))
      val tailLiterals = s.reshape()()(Map("data" -> literalMatrixEmbedding(tail), "shape" -> Shape(batchSize, 1, numRelations)))
      val corruptHeadLiterals = s.reshape()()(Map("data" -> literalMatrixEmbedding(corruptHead), "shape" -> Shape(batchSize, 1, numRelations)))
      val corruptTailLiterals = s.reshape()()(Map("data" -> literalMatrixEmbedding(corruptTail), "shape" -> Shape(batchSize, 1, numRelations)))

      head = entityEmbedding(head)
      relation = relationEmbedding(relation)
      tail = entityEmbedding(tail)
      corruptHead = entityEmbedding(corruptHead)
      corruptTail = entityEmbedding(corruptTail)

//      head = s.concat()(head, headLiterals)(Map("num_args" -> 2,  "dim" -> 2))
//      relation = s.concat()(relation, s.ones_like()()(Map("data" -> headLiterals)))(Map("num_args" -> 2,  "dim" -> 2))
//      tail = s.concat()(tail, tailLiterals)(Map("num_args" -> 2,  "dim" -> 2))
//      corruptHead = s.concat()(corruptHead, corruptHeadLiterals)(Map("num_args" -> 2,  "dim" -> 2))
//      corruptTail = s.concat()(corruptTail, corruptTailLiterals)(Map("num_args" -> 2,  "dim" -> 2))

      val posScore = getScore(head, headLiterals, relation, tail, tailLiterals)
      val negScore = getScore(corruptHead, corruptHeadLiterals, relation, corruptTail, corruptTailLiterals)

//      MaxMarginLoss(1.0f)(posScore, negScore)
      posScore
//      s.BlockGrad()()(Map("data" -> posScore))
    }

    // Literals model
//    val literalLoss = {
//      val input = s.Variable("literaldata")
//      val label = s.Variable("linearregressionoutput0_label")
//      val splitInput = s.split()()(Map("data" -> input, "axis" -> 1, "num_outputs" -> 2))
//      val literalHead = splitInput.get(0)
//      val literalRelation = splitInput.get(1)
//
//      val origHead = entityEmbedding(literalHead)
//      val origRelation = relationEmbedding(literalRelation)
//
//      var litHead = literalEntityEmbedding(literalHead)
//      litHead = s.Activation()()(Map("data" -> litHead, "act_type" -> "relu"))
//      litHead = s.FullyConnected()()(Map("data" -> litHead, "num_hidden" -> latentFactors*2))
//      litHead = s.FullyConnected()()(Map("data" -> litHead, "num_hidden" -> latentFactors))
//      litHead = litHead + s.reshape()()(Map("data" -> origHead, "shape" -> Shape(batchSize, latentFactors)))
//      litHead = s.Dropout()()(Map("data" -> litHead, "p" -> 0.5))
//
//      var litRelation = literalRelationEmbedding(literalRelation)
//      litRelation = s.Activation()()(Map("data" -> litRelation, "act_type" -> "relu"))
//      litRelation = s.FullyConnected()()(Map("data" -> litRelation, "num_hidden" -> latentFactors*2))
//      litRelation = s.FullyConnected()()(Map("data" -> litRelation, "num_hidden" -> latentFactors))
//      litRelation = litRelation + s.reshape()()(Map("data" -> origRelation, "shape" -> Shape(batchSize, latentFactors)))
//      litRelation = s.Dropout()()(Map("data" -> litRelation, "p" -> 0.5))
//
//      var pred = litHead * litRelation
//      pred = s.sum_axis()()(Map("data" -> pred, "axis" -> Shape(1)))
//      pred = s.Flatten()()(Map("data" -> pred))
//
//      val loss = s.LinearRegressionOutput()()(Map("data" ->  pred, "label" -> label))
//      loss
//    }


    (loss, score, Seq("data", "linearregressionoutput0_label"))
  }

  def train() = {
    val ctx = Context.gpu(0)
    val ctx2 = Context.gpu(0)
    //  val numEntities = 40943
    val (transeModel, scoreModel, paramNames) = getNet()

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
    println(literals.length)
        var literalMatrix = NDArray.array(SparseMatrix.fromCOO(numEntities, numRelations, literals).toArray.map(_.toFloat),
          Shape(numEntities, numRelations)).copyTo(ctx)
        val means = NDArray.mean(literalMatrix, 0)
        val sd = NDArray.sqrt(NDArray.mean(NDArray.square(NDArray.broadcast_minus(literalMatrix, means)), 0))
        literalMatrix = NDArray.broadcast_div(NDArray.broadcast_minus(literalMatrix, means), sd)
        val rowSums = NDArray.sum(literalMatrix, 1)
        literalMatrix = NDArray.broadcast_div(literalMatrix, rowSums.get.reshape(Shape(numEntities, 1))).copyTo(ctx)
//        val denseLiterals = DenseMatrix.create(numEntities, numRelations, literalMatrix.toArray)

//        var literalData = {
//          var literalCoordinates = literals.map{
//            case (row: Int, col: Int, _) =>
//              Seq(row.toFloat, col.toFloat)
//          }
//
//          literalCoordinates = literalCoordinates ++ literalCoordinates.take(batchSize - (literalCoordinates.size % batchSize))
//
//          literalCoordinates.grouped(batchSize)
//            .map(x => NDArray.array(x.flatten.toArray, Shape(batchSize, 2)))
//            .toSeq
//        }
//
//        var literalLabels = {
//          var literalValues = literals.map{
//            case (row, col, _) =>
//              denseLiterals(row, col)
//          }
//
//          literalValues = literalValues ++ literalValues.take(batchSize - (literalValues.size % batchSize))
//
//          literalValues.grouped(batchSize)
//            .map(x => NDArray.array(x.toArray, Shape(batchSize, 1)))
//            .toSeq
//        }

    //        val (testSubjects, testRelations, testObjects, _, _) = readDataBatched("test")

//        var literalAll = {
//          var literalCoordValues = literals.map{
//            case (row: Int, col: Int, value: Double) =>
//              Seq(row, col, value.toFloat)
//          }
//
//          literalCoordValues = literalCoordValues ++ literalCoordValues.take(batchSize - (literalCoordValues.size % batchSize))
//
//          literalCoordValues.grouped(batchSize)
//            .map(x => NDArray.array(x.flatten.toArray, Shape(batchSize, 3)))
//            .toSeq
//        }
//
//
//        if(inputStuff.length < literalAll.length) {
//          inputStuff = inputStuff ++ inputStuff.take(literalAll.length - inputStuff.length)
//        } else if(inputStuff.length > literalAll.length) {
//          literalAll = literalAll ++ literalAll.take(inputStuff.length - literalAll.length)
//        }

//        val finalData = inputStuff.zip(literalAll).map {
//          case (triples: NDArray, literals: NDArray) =>
//            NDArray.concat(triples, literals, 2, 1).get
//        }

//        if(inputStuff.length < literalData.length) {
//          inputStuff = inputStuff ++ inputStuff.take(literalData.length - inputStuff.length)
//        } else if(inputStuff.length > literalData.length) {
//          literalData = literalData ++ literalData.take(inputStuff.length - literalData.length)
//
//        }
//    186840
        println(inputStuff.size)
//    println(literalData.size)
//
//        val tripleShape = inputStuff.head.shape
//        val literalShape = literalData.head.shape
//
//        val finalInputStuff = inputStuff.zip(inputStuff.map(x => NDArray.zeros(literalShape))).map{
//          case (triples: NDArray, dummy: NDArray) =>
//            NDArray.concat(triples, dummy, 2, 1).get
//        }
//
//        println(finalInputStuff.head.shape)
//    println(finalInputStuff.size)
//
//        val finalLiteralAll = literalData.map(x => NDArray.zeros(tripleShape)).zip(literalData).map{
//          case (dummy: NDArray, literals: NDArray) =>
//            NDArray.concat(dummy, literals, 2, 1).get
//        }
//
//    println(finalLiteralAll.head.shape)

//        val finalData = finalInputStuff zip finalLiteralAll flatMap { case (a, b) => Seq(a, b) }

        var minTestHits = 300f
        val opt = new AdaGrad(learningRate = 0.01f)
//        println(literalMatrix.shape)
        val trainer = new Model(opt, s.Group(transeModel), scoreModel, inputStuff, null, paramNames, ctx, "data", null, literalMatrix)
//        val literalTrainer = new Model(opt, literalModel, scoreModel, literalData, paramNames, ctx, "literaldata", labels = Some(literalLabels))
        for (iter <- 0 until inputStuff.size * 10) {
          trainer.trainTripleBatch()
          println(s"E model: iter $iter, training loss: ${trainer.tripleExecutor.outputs(0).toArray.sum}")
//          println(s"E model: iter $iter, training loss: ${trainer.tripleExecutor.outputs(0).toArray.sum}, pos score: ${trainer.tripleExecutor.outputs(1).toArray.sum}, neg score: ${trainer.tripleExecutor.outputs(2).toArray.sum}")
//          trainer.trainLiteralBatch()
//          literalTrainer.trainBatch()
//                println(s"iter $epoch, training Hits@1: ${Math.sqrt(Hits.hitsAt1(NDArray.ones(batchSize), executor.outputs(0)) / batchSize)}, min test Hits@1: $minTestHits")
//          println(s"L model: iter $iter, training loss: ${trainer.tripleExecutor.outputs(1).toArray.sum}")
//          println(s"Literal model: iter $iter, training loss: ${trainer.literalExecutor.outputs(0).toArray.sum}")
        }


    val model2 = trainer.getFeedforwardModel
//
    val vectors = new Random(5).shuffle(readDataBatched("train", test = false).flatten.map{
      case x =>
        LabeledPoint(0.0, Vectors.dense(x.map(_.toDouble)))
    }).take(50000).map(_.features)
//    val dt = new MyPointIter(vectors.toIterator, Shape(5), batchSize, "data", "label", ctx)
//    println(dt.next().data.head.shape)
//    println(dt.next().data.size)
//    val predictions = model2.predict(dt).map(_.copyTo(ctx2))
//    println(model2.getArgParams.values.map(_.shape))
//    println(model2.getAuxParams.values.map(_.shape))
//      results.map(arr => MXNDArray(arr))
//      val predictions = model.predict(input.map(_.features).toIterator)

//    val hits = new EvalMetrics(model2, 1, batchSize,
//      NDArray.array(Array.range(0,numRelations).map(_.toFloat), Shape(1,numRelations), ctx2),
//      new MyPointIter(vectors.toIterator, Shape(5), batchSize, "data", "label", ctx2), ctx2)
//    println(hits.hits(5,10).mkString(","))
//    println(hits.mrr)

    val vectors2 = readDataBatched("test", test = false).flatten.map{
      case x =>
        LabeledPoint(0.0, Vectors.dense(x.map(_.toDouble)))
    }.map(_.features)

//    val predictions = model2.predict(new MyPointIter(vectors2.toIterator, Shape(5), batchSize, "data", "label", ctx2), batchSize)
//    println(predictions.map(x => x.toArray.mkString("|")).mkString("\n\n"))

//    val evalBatchSize = 10

    val hitsTest = new EvalMetrics(model2, 0, batchSize,
      NDArray.array(Array.range(0,numEntities).map(_.toFloat), Shape(1,numEntities), ctx2),
      new MyPointIter(vectors2.toIterator, Shape(5), batchSize, "data", "label", ctx2), ctx2)
    //    println(hitsTest.hits(5,10).mkString(","))
    println(hitsTest.mrr)

    val hitsTrain = new EvalMetrics(model2, 0, batchSize,
      NDArray.array(Array.range(0,numEntities).map(_.toFloat), Shape(1,numEntities), ctx2),
      new MyPointIter(vectors.toIterator, Shape(5), batchSize, "data", "label", ctx2), ctx2)
//    println(hitsTrain.hits(5,10).mkString(","))
    println(hitsTrain.mrr)


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
//    val triplesFile = s"/home/NileshChakraborty/Spark-Tensors/data/$stage.txt"
//    val entityIDFile = "/home/NileshChakraborty/Spark-Tensors/data/entity2id.txt"
//    val relationIDFile = "/home/NileshChakraborty/Spark-Tensors/data/relation2id.txt"

    val triplesFile = s"/home/NileshChakraborty/exp/$stage.txt"
    val entityIDFile = "/home/NileshChakraborty/exp/entity2id.txt"
    val relationIDFile = "/home/NileshChakraborty/exp/relationwl2id.txt"

    def getIDMap(path: String) = Source.fromFile(path)
      .getLines()
      .map(_.split("\t"))
      .map(x => x(0) -> x(1).toFloat).toMap

    val entityID = getIDMap(entityIDFile)
    val relationID = getIDMap(relationIDFile)

    val triples = Source.fromFile(triplesFile).getLines().map(_.split("\t")).toSeq
    val mappedTriples = triples.flatMap{
      case posTriple =>
        Seq.fill(1)(posTriple).map {
          case x =>
            Array(entityID(x(0)),
              relationID(x(1)),
              entityID(x(2))) ++
              (if (!test) {
                if(Random.nextInt(2) == 0)
                  Array(Random.nextInt(numEntities).toFloat,
                    entityID(x(2)))
                else
                  Array(entityID(x(0)),
                    Random.nextInt(numEntities).toFloat)
              }
              else Nil)
        }
    }


    Random.shuffle(mappedTriples ++ mappedTriples.take(batchSize - (mappedTriples.size % batchSize)))
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
    val literalsFile = s"/home/NileshChakraborty/exp/literals.txt"
    val entityIDFile = "/home/NileshChakraborty/exp/entity2id.txt"
    val relationIDFile = "/home/NileshChakraborty/exp/relationwl2id.txt"

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
          case ex: NumberFormatException =>
            Nil
          case ex: ArrayIndexOutOfBoundsException =>
            Nil
        }
    }

    Random.shuffle(mappedTriples)
  }
}