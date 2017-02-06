import pyspark
import pyspark.mllib.linalg
import pyspark.ml.param
from pyspark.ml.param.shared import *

@pyspark.mllib.common.inherit_doc
class PairwiseEuclideanDistance(pyspark.ml.pipeline.Transformer,
                                HasInputCol,
                                HasOutputCol):
    """
    A transformer that takes a dataframe with a feature column as vectors, and
    compute the Euclidean distance matrix between each pair of vectors.
    """
    # a placeholder to make it appear in the generated doc
    squared = pyspark.ml.param.Param(Params._dummy(), "squared", 
                            "boolean to indicate squared Euclidean distance")

    @pyspark.ml.util.keyword_only
    def __init__(self, 
                 squared=False,
                 inputCol="feature",
                 outputCol="distance"):
        """
        __init__(self, squared=False, inputCol="feature", outputCol="distance")
        """
        super(PairwiseEuclideanDistance, self).__init__()
        self.squared = pyspark.ml.param.Param(self, "squared", 
                            "boolean to indicate squared Euclidean distance")
        self._setDefault(squared=False, inputCol="feature", 
                            outputCol="distance")
        kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)
    
    @pyspark.ml.util.keyword_only
    def setParams(self, 
                  squared=False,
                  inputCol="feature",
                  outputCol="distance"):
        """
        setParams(self, squared=False, inputCol="feature", outputCol="distance")
        Sets params for this EuclideanPairwiseDistance.
        """
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)
    
    def setSquared(self, value):
        """
        Sets the value of :py:attr:`squared`.
        """
        self._paramMap[self.squared] = value
        return self

    def getSquared(self):
        """
        Gets the value of squared or its default value.
        """
        return self.getOrDefault(self.squared)
    
    def _transform(self, dataset):
        """
        Transforms the input dataset.
        """
        rdd = dataset.select(self.getInputCol()).map(lambda r: r[0])
        rdd = rdd.zipWithIndex()
        df = rdd.toDF()
        # Perform Cartesian Product of dataframe 
        df = df.join(df)
        if self.getSquared():
            # Squared Euclidean distance
            rdd = df.map(lambda r: [r[1], r[0], r[3], r[2],
                                   float(r[0].squared_distance(r[2]))])
        else:
            # Euclidean distance
            rdd = df.map(lambda r: [r[1], r[0], r[3], r[2],
                                   float(r[0].squared_distance(r[2]))**0.5])
        df = rdd.toDF(["x_id", "x_"+self.getInputCol(),
                       "y_id", "y_"+self.getInputCol(), self.getOutputCol()])
        return df

def _test():
    """
    Unit test function for PairwiseEuclideanDistance class.
    """
    try:
        if isinstance(sc, pyspark.SparkContext):
            print("SparkContext:", sc)
    except Exception as ex:
        sc = pyspark.SparkContext()
        print("Created SparkContext:", sc)
    try:
        if isinstance(sqlContext, pyspark.SQLContext):
            print("SQLContext:", sqlContext)
    except Exception as ex:
        sqlContext = pyspark.SQLContext(sc)
        print("Created SQLContext:", sqlContext)

    # Generate dummy vectors
    rdd_data = [(pyspark.mllib.linalg.Vectors.dense([0.0, 1.0]),),
                (pyspark.mllib.linalg.Vectors.dense([1.0, 1.0]),),
                (pyspark.mllib.linalg.Vectors.dense([2.0, 3.0]),),
                (pyspark.mllib.linalg.Vectors.dense([8.0, 9.0]),)]
    df_data = sqlContext.createDataFrame(rdd_data, ["feature"])
    df_data.show()
    df_data.collect()

    # Instantiate a pairwise Euclidean distance transformer
    ped = PairwiseEuclideanDistance(squared=False,
                                    inputCol="feature",
                                    outputCol="distance")
    
    # Get squared Euclidean pairwise distances
    ped.setSquared(True)
    ped.getSquared()
    ped.getInputCol()
    ped.hasDefault("squared")
    print(ped.explainParams())
    ped.transform(df_data).show()

    # Get Euclidean pairwise distances
    ped.setSquared(False)
    print(ped.explainParams())
    ped.transform(df_data).show()

    # Shutdown SparkContext
    sc.stop()
    print("SparkContext is shutdown.")

if __name__ == "__main__":
    _test()
