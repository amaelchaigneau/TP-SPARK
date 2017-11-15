package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.feature.IDF
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.tuning.TrainValidationSplit
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.tuning.TrainValidationSplitModel

object Predict {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"))

    val spark = SparkSession
      .builder
      .config(conf)
      .master("local")
      .appName("TP_spark")
      .getOrCreate()

    var dir = "/home/amael/mastere-telecom/cours/spark/tp/"

    var modelFilename = dir + "model"

    var filename = dir + "train-filtered"

    var df = spark.read.format("parquet").load(dir + "/df")
    var df_train = spark.read.format("parquet").load(dir + "/df_train")
    var df_test = spark.read.format("parquet").load(dir + "/df_test")

    var model = TrainValidationSplitModel.load(dir + "/model")
    
    var df_WithPredictions = model.transform(df_test)

    df_WithPredictions.groupBy("final_status", "prediction").count().show

  }
}
