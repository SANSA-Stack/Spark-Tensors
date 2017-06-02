package net.sansa_stack.ml.kge.model

import ml.dmlc.mxnet._
import ml.dmlc.mxnet.{Symbol => s}

/**
  * Created by nilesh on 01/06/2017.
  */
class TransE(numEntities: Long, numRelations: Long, latentFactors: Long) {
  val entityEmbeddings =
    s.Embedding("entity")()(Map("name" -> "embed", "input_dim" -> numEntities, "output_dim" -> latentFactors))
  val relationEmbeddings =
    s.Embedding("entity")()(Map("name" -> "embed", "input_dim" -> numEntities, "output_dim" -> latentFactors))

  def score() = {
    
  }
}
