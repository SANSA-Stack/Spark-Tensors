package tensor.ml.kge.dataset.rdd


import org.apache.spark._
import org.apache.spark.rdd._
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import scala.util.Random
import org.apache.spark.sql.functions._
import org.apache.spark.sql.catalyst.encoders.RowEncoder

//case class Record(Subject: String, Predicate:String, Object:String)



class RDDTriples ( name: String,
		               spark : SparkSession, 
		               filePathTriples : String) {


	import spark.implicits._  // to be able to work with """spark.sql("blah blah").as[String].rdd"""

	type rowType = (String, String, String)
	var triples : RDD[rowType] 	= readFromFile()


	def readFromFile(delimiter:String = "\t", header:Boolean = false) : RDD[rowType] = {

			triples = spark.read.text(filePathTriples).as[String]
	              .map{ line:String => line.split(delimiter) }
	              .map{ case Array(s:String, p:String, o:String, _*) => (s.trim(), p.trim(), o.trim() ) }
	              .rdd
	    
	    return triples
	}

	def getAllDistinctEntities() : RDD[String] = {
	  
	  return triples.map{ spo : rowType => spo._1 }
	                .union( triples.map{ spo : rowType => spo._3 } )
	                .distinct()
	}

	def getAllDistinctPredicates() : RDD[String] = {

			return triples.map{ spo : rowType => spo._2 }
			              .distinct()
	}


	def corruptSubjectOrObject(rddtriples : RDD[rowType],
	                           probabilityToMutateSubjectWithRespectToObject : Double = 0.5) : RDD[rowType] = {
					
//	  val x =  rddtriples.map{
//	     trp =>
//	       if ( Random.nextDouble() < probabilityToMutateSubjectWithRespectToObject )
//	       {
	         // mutating subject 
//	         triples.filter{
//	           spo => 
//	             val t=((spo._1 != trp._1) &&
//	              (spo._2 == trp._2) &&
//	              (spo._3 == trp._3))
//	              println("trp=",trp.toString(),"spo=",spo.toString(),"t=",t.toString())
//	         }.sample(false, .3)
//	          .first() 
//	       } 	     
	     
//	   }
		
	  
//	  val x = rddtriples.map{
//	    trp =>
//	      println(trp)
//
//	      trp
//	  }
	      
//	  	   triples.filter{
//	        spo =>
//	          if(spo._1 != "aaa") println("yes") else println("no")       
//	          false
//	      }.collect()
	  
	  val checkIt = (trp1 : rowType,trp2: rowType) => {
	    (trp1._1 == trp2._1) && (trp1._2 == trp2._2) && (trp1._3 == trp2._3)
	  }  
	  
	  val x = rddtriples.flatMap{
	    trp : rowType => {
	      val alltriples = triples
	      alltriples.filter{
	        spo : rowType =>
	          true
	      }.asInstanceOf[List[rowType]]
	    }
	  }
	  
	  
	  
	  x.take(1).foreach(println)
		 return null
	}




}