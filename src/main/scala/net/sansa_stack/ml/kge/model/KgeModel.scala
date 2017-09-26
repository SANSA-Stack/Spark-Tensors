//package net.sansa_stack.ml.kge.model
//
//import ml.dmlc.mxnet.module.Module
//import ml.dmlc.mxnet.{Symbol => s, _}
//import ml.dmlc.mxnet.Symbol
//import ml.dmlc.mxnet.optimizer.AdaGrad
//import ml.dmlc.mxnet.spark.io.MyPointIter
//import org.apache.spark.mllib.linalg.Vectors
//import org.apache.spark.mllib.regression.LabeledPoint
//
//import scala.util.Random
//
///**
//  * Created by nilesh on 13/09/2017.
//  */
//trait KgeModel {
//  val numEntities: Int
//  val numRelations: Int
//  val latentFactors: Int
//  val trainBatchSize: Int
//  val testBatchSize: Int
//
//  protected val entityWeight = s.Variable("entity_weight")
//  protected val relationWeight = s.Variable("relation_weight")
//
//  private val input = s.split()()(Map("data" -> s.Variable("data"), "axis" -> 1, "num_outputs" -> 5))
//  protected val head = entityEmbedding(input.get(0))
//  protected val relation = relationEmbedding(input.get(1))
//  protected val tail = entityEmbedding(input.get(2))
//  protected val corruptHead = entityEmbedding(input.get(3))
//  protected val corruptTail = entityEmbedding(input.get(4))
//
//  protected def entityEmbedding(id: Symbol): Symbol
//  protected def relationEmbedding(id: Symbol): Symbol
//
//  def getLoss: Symbol
//  def getScore: Symbol
//}
//
//trait NegativeSampler {
//  val model: KgeModel
//
//  protected def corruptSample(s: Int, p: Int, o: Int): (Int, Int, Int) = {
//    if(Random.nextInt(2) == 0)
//      (Random.nextInt(model.numEntities), p, o)
//    else
//      (s, p, Random.nextInt(model.numEntities))
//  }
//}
//
//class ModelTrainer(val model: KgeModel, val name: String) extends NegativeSampler {
//  val context = Context.gpu()
//  private val trainModule = new Module(model.getLoss, contexts=context)
//  private val testModule = new Module(model.getScore, contexts=context)
//
//  def fit(trainData: DataIter,
//          validationData: Option[DataIter] = None,
//          numExamples: Int,
//          batchSize: Int,
//          learningRate: Float,
//          weightDecay: Float,
//          devs: Array[Context],
//          earlyStopping: Boolean = true) = {
//    val checkpoint = new EpochEndCallback {
//        override def invoke(epoch: Int, symbol: Symbol,
//                            argParams: Map[String, NDArray],
//                            auxParams: Map[String, NDArray]): Unit = {
//          Model.saveCheckpoint(name, epoch + 1, symbol, argParams, auxParams)
//        }
//      }
//
//    val epochSize = numExamples / batchSize
//
//    trainModule.bind(trainData.provideData)
//    trainModule.initParams(new Xavier(factorType = "in", magnitude = 2.34f))
//    trainModule.initOptimizer(optimizer = new AdaGrad(learningRate = learningRate, wd = weightDecay))
//
//    println(epochSize)
//
//    for (iter <- 0 until epochSize * 1) {
//      if (!trainData.hasNext || (iter > 0 && iter % epochSize == 0)) {
//        println("Epoch end!")
//        trainData.reset()
//      }
//      trainModule.forwardBackward(trainData.next())
//      trainModule.update()
//      println(s"E model: iter $iter, training loss: ${trainModule.getOutputsMerged()(0).toArray.sum}")
//    }
////    trainModule.fit(trainData)
//  }
//
//  def setupTester(data: DataIter): Unit = {
//    val (argParams, auxParams) = trainModule.getParams
//    testModule.bind(data.provideData)
//    testModule.initParams(argParams = argParams, auxParams = auxParams, allowMissing = true)
//  }
//
//  def predict(data: DataIter): IndexedSeq[NDArray] = {
//    testModule.predict(data, model.testBatchSize)
//  }
//}