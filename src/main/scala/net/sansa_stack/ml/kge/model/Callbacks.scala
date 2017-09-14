package net.sansa_stack.ml.kge.model

import ml.dmlc.mxnet.{EpochEndCallback, FeedForward, NDArray, Symbol}

/**
  * Created by nilesh on 07/07/2017.
  */
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
