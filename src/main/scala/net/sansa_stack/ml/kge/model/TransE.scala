package net.sansa_stack.ml.kge.model


import ml.dmlc.mxnet.{Symbol, Symbol => s}
import net.sansa_stack.ml.kge.{L1Similarity, MaxMarginLoss}

/**
  * Created by nilesh on 14/09/2017.
  */
class TransE(val numEntities: Int,
             val numRelations: Int,
             val latentFactors: Int,
             val trainBatchSize: Int,
             val testBatchSize: Int) extends KgeModel {

  protected def entityEmbedding(data: Symbol) =
    s.Embedding("entities")()(Map("data" -> data, "weight" -> entityWeight, "input_dim" -> numEntities, "output_dim" -> latentFactors))

  protected def relationEmbedding(data: Symbol) =
    s.Embedding("relations")()(Map("data" -> data, "weight" -> relationWeight, "input_dim" -> numRelations, "output_dim" -> latentFactors))

  protected def getScore(head: Symbol, relation: Symbol, tail: Symbol) = {
    L1Similarity(head + relation, tail)
  }

  protected val posScore = getScore(head, relation, tail)
  protected val negScore = getScore(corruptHead, relation, corruptTail)

  def getLoss: Symbol = MaxMarginLoss(1.0f)(posScore, negScore)

  def getScore: Symbol = posScore
}
