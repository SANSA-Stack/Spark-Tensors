package tensor.ml.kge.dataset.dataframe

import org.apache.spark._
import org.apache.spark.rdd._
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import scala.util.Random
import org.apache.spark.sql.functions._
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import com.twitter.chill._

case class Record(Subject: String, Predicate:String, Object:String)

class Triples ( name: String,
		spark : SparkSession, 
		filePathTriples : String) {


	import spark.implicits._  // to be able to work with """spark.sql("blah blah").as[String].rdd"""

	val schema = StructType(Array(
			StructField("Subject", StringType, true),
			StructField("Predicate", StringType, true),
			StructField("Object", StringType, true) ))
			
	implicit 	val encoder = RowEncoder(schema)

//  case class Record(Subject: String, Predicate:String, Object:String)

	var triples : DataFrame = readFromFile()

	val nameViewTriples = name + "OriginalTriples"

	triples.createOrReplaceTempView(nameViewTriples)



	def readFromFile(delimiter:String = "\t", header:Boolean = false):DataFrame = {

			triples = spark.read.format("com.databricks.spark.csv")
					.option("header", header.toString() )
					.option("inferSchema",false)
					.option("delimiter", delimiter)
					.schema(schema)
					.load(filePathTriples)

					// triples = triples.withColumn("ID", monotonicallyIncreasingId())  // augmenting with an uinque ID column
			return triples

	}

	def encodeOriginalTriplesToNumeric():DataFrame = {
			println("Implement Me! - encodeOriginalTriplesToNumeric")
			return null
	}

	def getAllDistinctEntities() : RDD[String] = {


			val query = "SELECT Subject FROM "+nameViewTriples+" UNION SELECT Object FROM "+nameViewTriples

			return spark.sql(query).as[String].rdd 
	}

	def getAllDistinctPredicates() : RDD[String] = {

			return triples.select("Predicate").distinct().as[String].rdd
	}

	def corruptSubjectOrObject2(dfTriples : DataFrame,
	                           probabilityToMutateSubjectWithRespectToObject : Double = 0.5) : DataFrame = {
	  
  	dfTriples.createOrReplaceTempView("dfTriples")
  	
  	val query = " SELECT main.Subject, main.Predicate, main.Object " + 
  	            " FROM " + nameViewTriples + """ AS main
  	              INNER JOIN dfTriples AS df 
  	              WHERE df.Subject != main.Subject AND
  	                    df.Predicate = main.Predicate AND
  	                    df.Object  = main.Object
  	          """
  	            
  	return spark.sql(query) 

	  return null
	}
	
	def fun1(dfTriples : DataFrame,
	         probabilityToMutateSubjectWithRespectToObject : Double = 0.5) : DataFrame = {
	  
	  
	 val name = "maintriples"
	 val trp = triples
	 
	 trp.createOrReplaceTempView(name)

	 com.twitter.chill.MeatLocker
	 
	 return  dfTriples.map{
	    row =>
	      val q = "SELECT Subject FROM " + name
	      
	      spark.sql(q).take(1).asInstanceOf[String]
	      
	      row
	  }
	  
	  return null
	}
	
	
	def corruptSubjectOrObject(dfTriples : DataFrame,
	                           probabilityToMutateSubjectWithRespectToObject : Double = 0.5) : DataFrame = {
					

	    import spark.implicits._
	    import org.apache.spark.sql.Row

			var entityToMutate : String = ""
			var selectedToMutate : String = ""
			var ds : Dataset[Row]  = null

//			val x = dfTriples.as[Record].map{
//	      case row => row // triples.filter($"Subject" === row.Subject)
//	    }.asInstanceOf[DataFrame]
//	
			
//		  val x = dfTriples.as[Record].flatMap{
//	      case row =>  triples.filter($"Subject" === row.Subject).asInstanceOf[List[Record]]
//	    }.asInstanceOf[DataFrame]
//	
			  

//			val x = dfTriples.as[Record].map{
//	      case row => triples.filter($"Subject" === row.Subject)(encoder)
//	    }.asInstanceOf[DataFrame]
			
//			val x = dfTriples.as[Record].map{
//	      case row => ("key",triples.filter($"Subject" === row.Subject) )
//	    }.rdd.reduceByKey((a,b)=>a.union(b)).values.first()
//	    
//	    val x= triples.filter($"Subject" === "sss").sample(true, 0.3)
	    
	    
//val x = triples.filter($"Subject" === "2")
	    
	    
//			val x = dfTriples.as[Record].map{
//	      case row => 
//	        if (Random.nextDouble() < probabilityToMutateSubjectWithRespectToObject ) {
//	          entityToMutate = row.Subject
//	          	          
//	          triples.except(
//	              triples.filter(
//	                  ($"Subject" !== row.Subject) &&
//	                  ($"Predicate" === row.Predicate) &&
//	                  ($"Object" === row.Object) )
//                 .distinct() )
//               .sample(false, 0.5).first()
//	          
//	        } // else {
//	    }//.asInstanceOf[Dataset[Row]]

//			val x = dfTriples.map{
//	      case Row(s:String, p:String, o:String) => 
//    				if (Random.nextDouble() < probabilityToMutateSubjectWithRespectToObject ) {
//    					selectedToMutate = "Subject"
//    							entityToMutate = s
//    
//    							triples.except(triples.filter($"Subjects" !== entityToMutate).distinct() ).sample(false, 0.5).first()
//    
//    				} else {
//    					selectedToMutate = "Object"
//    							entityToMutate = o
//    
//    							triples.except(triples.filter($"Objects" !== entityToMutate).distinct() ).sample(false, 0.5).first()
//    				}
//			}.asInstanceOf[Dataset[Row]]
			
			return null
	}

	def setColumnNames(df:DataFrame, names:Seq[String] = Seq("Subject","Predicate","Object","ID") ):DataFrame = {
			// using: https://stackoverflow.com/questions/35592917/renaming-column-names-of-a-data-frame-in-spark-scala

			df.toDF(names: _*)
	}


}