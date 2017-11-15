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
import org.apache.spark.sql.SaveMode

object Trainer {

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

    /**
     * *****************************************************************************
     *
     *       TP 4-5
     *
     *       - lire le fichier sauvegarder précédemment
     *       - construire les Stages du pipeline, puis les assembler
     *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
     *       - Sauvegarder le pipeline entraîné
     *
     *       if problems with unimported modules => sbt plugins update
     *
     * ******************************************************************************
     */

    /** CHARGER LE DATASET **/

    var dir = "/home/amael/mastere-telecom/cours/spark/tp/ser/"

    var filename = "/home/amael/mastere-telecom/cours/spark/tp/" + "train-filtered"

    var df = spark.read.format("parquet").load(filename)

    /** TF-IDF **/

    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    var wordRemover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("words")

    val countVectorizer = new CountVectorizer()
      .setInputCol("tokens")
      .setOutputCol("tokens2")

    val idf = new IDF().setInputCol("tokens2").setOutputCol("tfidf")

    /** handleInvalid à skip pour régler le problème des labels rencontrés dans le dataset de test et non dans le dataset de training **/
    
    var countryIndexer = new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("country")
      .setOutputCol("country_indexed")

    var currencyIndexer = new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("currency")
      .setOutputCol("currency_indexed")

    /** VECTOR ASSEMBLER **/

    var vectAssemblor = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_indexed", "currency_indexed", "hours_prepa"))
      .setOutputCol("features")

    /** MODEL **/

    val lr = new LogisticRegression()
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("prediction")
      .setRawPredictionCol("raw_prediction")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)

    /** PIPELINE **/

    val pipeLine = new Pipeline()
      .setStages(Array(tokenizer, wordRemover, countVectorizer, idf, countryIndexer, currencyIndexer, vectAssemblor, lr))

    pipeLine.write.overwrite().save(dir + "/pipeline")

    /** Split df **/

    var df_tab = df.randomSplit(Array(0.1, 0.9))

    var df_test = df_tab(0)

    var df_train = df_tab(1)

    df.write.mode(SaveMode.Overwrite).format("parquet").save(dir + "/df");
    df_test.write.mode(SaveMode.Overwrite).format("parquet").save(dir + "/df_test");
    df_train.write.mode(SaveMode.Overwrite).format("parquet").save(dir + "/df_train");

    /** TRAINING AND GRID-SEARCH **/

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.elasticNetParam, Array(1e-08, 1e-06, 1e-04, 1e-02))
      .addGrid(countVectorizer.minDF, Array(55.0, 75.0, 95.0))
      .build()

    var evaluator = new MulticlassClassificationEvaluator().setMetricName("f1").setLabelCol("final_status").setPredictionCol("prediction")

    var gs = new TrainValidationSplit().setEstimator(pipeLine).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setTrainRatio(0.7)

    var model = gs.fit(df_train)

    model.write.overwrite().save(dir + "/model")

    var df_withPredictions = model.bestModel.transform(df_test)

    df_withPredictions.write.mode(SaveMode.Overwrite).format("parquet").save(dir + "/prediction")

    df_withPredictions.groupBy("final_status", "prediction").count().show

    println("f1 score:" + evaluator.evaluate(df_withPredictions))
  }
}
