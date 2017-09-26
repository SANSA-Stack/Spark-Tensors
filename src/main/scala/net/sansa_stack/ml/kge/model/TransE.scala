//package ml.dmlc.mxnet.spark.io
//
//
//import ml.dmlc.mxnet.spark.io.MyPointIter
//import ml.dmlc.mxnet.{Context, NDArray, Shape, Symbol}
//import ml.dmlc.mxnet.{Symbol => s}
//import net.sansa_stack.ml.kge.model.{KgeModel, ModelTrainer}
//import net.sansa_stack.ml.kge.{EvalMetrics, L1Similarity, MaxMarginLoss}
//import org.apache.spark.mllib.linalg.Vectors
//import org.apache.spark.mllib.regression.LabeledPoint
//
//import scala.io.Source
//import scala.util.Random
//
///**
//  * Created by nilesh on 14/09/2017.
//  */
//class TransE(val numEntities: Int,
//              val numRelations: Int,
//              val latentFactors: Int,
//              val trainBatchSize: Int,
//              val testBatchSize: Int) extends KgeModel {
//
//  protected def entityEmbedding(data: Symbol) =
//    s.Embedding("entities")()(Map("data" -> data, "weight" -> entityWeight, "input_dim" -> numEntities, "output_dim" -> latentFactors))
//
//  protected def relationEmbedding(data: Symbol) =
//    s.Embedding("relations")()(Map("data" -> data, "weight" -> relationWeight, "input_dim" -> numRelations, "output_dim" -> latentFactors))
//
//  protected def getScore(head: Symbol, relation: Symbol, tail: Symbol) = {
//    L1Similarity(head + relation, tail)
//  }
//
//  protected val posScore = getScore(head, relation, tail)
//  protected val negScore = getScore(corruptHead, relation, corruptTail)
//
//  def getLoss: Symbol = MaxMarginLoss(1.0f)(posScore, negScore)
//
//  def getScore: Symbol = posScore
//}
//
//object Main {
//  val model = new TransE(40943, 18, 100, 14200, 100)
//  val context = Context.gpu()
//
//  def main(args: Array[String]) {
//    val inputStuff = readDataBatched("train").flatten.map{
//      case x =>
//        LabeledPoint(0.0, Vectors.dense(x.map(_.toDouble)))
//    }.map(_.features)
//
//    val dataIter =  new MyPointIter(inputStuff.toIterator, Shape(5), model.trainBatchSize, "data", "label", context)
//
//    val vectors2 = readDataBatched("test", test = true).flatten.map{
//      case x =>
//        LabeledPoint(0.0, Vectors.dense(x.map(_.toDouble)))
//    }.map(_.features)
//
//    val myIter = new MyPointIter(vectors2.toIterator, Shape(5), model.testBatchSize, "data", "label", context)
//
//    val trainer = new ModelTrainer(model, "TransE")
//    trainer.fit(dataIter,
//      None,
//      inputStuff.size,
//      model.trainBatchSize,
//      0.1f,
//      0.0001f,
//      Context.gpu())
//    trainer.setupTester(myIter)
//
//    val numEntities2 = model.numEntities + (model.testBatchSize - model.numEntities % model.testBatchSize)
//
//    val hitsTest = new EvalMetrics(trainer, 0, model.testBatchSize,
//      NDArray.array(Array.range(0,numEntities2).map(_.toFloat), Shape(1,numEntities2), context),
//      myIter, context)
//    //    println(hits2.hits(5,10).mkString(","))
//    println(hitsTest.hits(10))
//
////    trainer.fit()
//  }
//
//  def readDataBatched(stage: String, test: Boolean = false) = {
//    val triplesFile = s"/home/NileshChakraborty/Spark-Tensors/data/$stage.txt"
//    val entityIDFile = "/home/NileshChakraborty/Spark-Tensors/data/entity2id.txt"
//    val relationIDFile = "/home/NileshChakraborty/Spark-Tensors/data/relation2id.txt"
//    //      val triplesFile = s"/data/nilesh/Spark-Tensors/data/$stage.txt"
//    //      val entityIDFile = "/data/nilesh/Spark-Tensors/data/entity2id.txt"
//    //      val relationIDFile = "/data/nilesh/Spark-Tensors/data/relation2id.txt"
//    //    val triplesFile = s"/home/NileshChakraborty/exp/$stage.txt"
//    //    val entityIDFile = "/home/NileshChakraborty/exp/entity2id.txt"
//    //    val relationIDFile = "/home/NileshChakraborty/exp/relation2id.txt"
//
//    def getIDMap(path: String) = Source.fromFile(path)
//      .getLines()
//      .map(_.split("\t"))
//      .map(x => x(0) -> x(1).toFloat).toMap
//
//    val entityID = getIDMap(entityIDFile)
//    val relationID = getIDMap(relationIDFile)
//
//    val triples = Source.fromFile(triplesFile).getLines().map(_.split("\t")).toSeq
//    val mappedTriples = triples.flatMap {
//      case posTriple =>
//        Seq.fill(1)(posTriple).map {
//          case x =>
//            Array(entityID(x(0)),
//              relationID(x(1)),
//              entityID(x(2))) ++
//              (if (Random.nextInt(2) == 0)
//                Array(Random.nextInt(model.numEntities).toFloat,
//                  entityID(x(2)))
//              else
//                Array(entityID(x(0)),
//                  Random.nextInt(model.numEntities).toFloat))
//        }
//    }
//
//    val bs = if (test) model.testBatchSize else model.trainBatchSize
//
//    Random.shuffle(mappedTriples ++ (if (mappedTriples.size % bs == 0) Nil else mappedTriples.take(bs - (mappedTriples.size % bs))))
//      .grouped(bs).toSeq
//  }
//}