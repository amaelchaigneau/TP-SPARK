package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import java.lang.Boolean
import org.apache.spark.sql.SaveMode
import org.apache.spark.sql.functions._

object Preprocessor {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .master("local")
      .getOrCreate()

    import spark.implicits._

    val sc = spark.sparkContext

    /**
     * *****************************************************************************
     *
     *       TP 2-3
     *
     *       - Charger un fichier csv dans un dataFrame
     *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
     *       - Sauver le dataframe au format parquet
     *
     *       if problems with unimported modules => sbt plugins update
     *
     * ******************************************************************************
     */

    var dir = "/home/amael/mastere-telecom/cours/spark/tp/tp2_3/TP_ParisTech_2017_2018_starter/data/"

    //    var filename = dir+"train.csv_replaced"
    var filename = dir + "train.csv"

    /** 1 - CHARGEMENT DES DONNEES **/

    var df = spark.read.format("csv")
      .option("header", "true")
      .load(filename)

    df = df.withColumn("goal", df.col("goal").cast("long"))
      .withColumn("backers_count", df.col("backers_count").cast("long"))
      .withColumn("final_status", df.col("final_status").cast("long"))

    df.printSchema()

    /** 2 - CLEANING **/

    df = df.filter($"final_status" === 1 or $"final_status" === 0)
    df = df.filter($"goal".isNotNull)
    df = df.drop("disable_communication")
    df = df.drop("state_changed_at", "backers_count")
    df = df.dropDuplicates("project_id")

    import java.sql.Timestamp

    def convertToTimestamp: (Long => Timestamp) = {
      i => new Timestamp(1000 * i)
    }

    def convertToBoolean: (String => Boolean) = {
      s => Boolean.parseBoolean(s)
    }

    import org.apache.spark.sql.functions.udf
    val udf_convertToTimestamp = udf(convertToTimestamp)

    val udf_convertToBoolean = udf(convertToBoolean)

    df = df.withColumn("created_at", udf_convertToTimestamp(df("created_at")))
    df = df.withColumn("launched_at", udf_convertToTimestamp(df("launched_at")))
    df = df.withColumn("deadline", udf_convertToTimestamp(df("deadline")))

    /** FEATURE ENGINEERING: Ajouter et manipuler des colonnes **/

    // days_campaign

    def computeDifference(t1: Timestamp, t2: Timestamp): Long = {
      var d: Long = 0
      if (t1 != null && t2 != null) {
        d = t2.getTime - t1.getTime
      } else {
        d = 0
      }
      d / 1000 / 3600 / 24
    }
    val differenceUDF = udf(computeDifference(_: Timestamp, _: Timestamp))

    def differenceHeures(t1: Timestamp, t2: Timestamp): Double = {
      if (t1 != null && t2 != null) {
        val d = t2.getTime - t1.getTime
        Math.round(d / 3600) / 1000
      } else {
        0
      }
    }
    val differenceHeuresUDF = udf(differenceHeures(_: Timestamp, _: Timestamp))

    df = df.withColumn("days_campaign", differenceUDF(df("launched_at"), df("deadline")))
    df = df.withColumn("hours_prepa", differenceHeuresUDF(df("created_at"), df("launched_at")))

    df = df.drop("deadline").drop("created_at").drop("launched_at")
    
    df = df.withColumn("name", lower($"name")).withColumn("desc", lower($"desc")).withColumn("keywords", lower($"keywords"))

    def concateneCols(s1: String, s2: String, s3: String): String = {
      var sf = s1 + " " + s2 + " " + s3
      val pattern = "\"+".r
      sf = pattern replaceFirstIn (sf, "")
      return sf
    }

    val concateneColsUDF = udf(concateneCols(_: String, _: String, _: String))

    df = df.withColumn("text", concateneColsUDF(df("name"), df("desc"), df("keywords")))

    df.toDF().show();
    
    df.write.mode(SaveMode.Overwrite).format("parquet").save("/home/amael/mastere-telecom/cours/spark/tp/train-filtered")

  }

}
