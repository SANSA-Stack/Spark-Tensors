package net.sansa_stack.ml.kge.model

import ml.dmlc.mxnet.{EpochEndCallback, NDArray}

/**
  * Created by nilesh on 7/7/17.
  */
trait KgeModel {
  def predictionModel(head: Symbol, relation: Symbol, tail: Symbol): Symbol
  def trainingModel(head: Symbol, relation: Symbol, tail: Symbol,
                    corruptedHead: Symbol, corruptedTail: Symbol): Symbol
  def train(lr: Float, )
}

object Callbacks {
  class KgeEpochEnd(validationData: NDArray) extends EpochEndCallback {
    override def invoke(epoch: Int, symbol: Symbol,
                        argParams: Map[String, NDArray],
                        auxStates: Map[String, NDArray]): Unit = {
      val numEntities = argParams("entity_weight").shape(0)
      val numRelations = argParams("relation_weight").shape(0)


    }
  }
}
