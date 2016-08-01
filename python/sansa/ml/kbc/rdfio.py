
import os
import rdflib
from rdflib.term import URIRef
import math
from pyspark import SQLContext, SparkContext, RDD
from pyspark.rdd import Partitioner

__author__ = 'nilesh'


class ThreeWayTensorPartitioner(Partitioner):
    def __init__(self, dimensions: tuple, blockSizes: tuple):
        self.dims = dimensions
        self.partitionSizes = blockSizes
        self.numPartitions = [int(math.ceil(self.dims[i] * 1.0 / self.partitionSizes[i])) for i in range(3)]
        self.totalPartitions = reduce(lambda x, y: x*y, self.numPartitions)

    def __eq__(self, other):
        return (isinstance(other, ThreeWayTensorPartitioner)
                and self.dims == other.dims
                and self.partitionSizes == other.partitionSizes)

    def __call__(self, k):
        return self.partitionFunc(k)

    def partitionFunc(self, key):
        for i in range(len(self.dims)):
            assert(0 <= key[i] <= self.dims[i])

        i, j, k = key
        ps1, ps2, ps3 = self.partitionSizes
        pn1, pn2, pn3 = self.numPartitions

        return i / ps1 + j / ps2 * pn1 + k / ps3 * pn2 * pn1


class RDFReader(object):
    def __init__(self, sc: SparkContext):
        self.sc = SQLContext(sc)

    def tripleRDD(self, file) -> RDD:
        def parseNTriples(lines):
            g = rdflib.Graph()
            g.parse(data="\n".join(lines), format="nt")
            allURIs = lambda statement: False not in [isinstance(term, URIRef) for term in statement]
            return [statement for statement in g if not allURIs(statement)]

        triples = self.sc.read.text(file).map(lambda x: x.value).mapPartitions(parseNTriples)
        return triples

    def tripleTensor(self, file, blockSizes: tuple):
        spo = self.tripleRDD(file)
        # Already filtered by URIs, no need to check types a la pattern matching
        entityIDs = spo.flatMap(lambda x: [x[0], x[2]]).distinct().zipWithUniqueId() # (eURI, eID)
        numEntities = entityIDs.countByKey()
        relationIDs = spo.map(lambda x: x[1]).distinct().zipWithUniqueId() # (rURI, rID)
        numRelations = relationIDs.countByKey()

        s_po = spo.map(lambda x: (x[0], (x[1], x[2])))

        def mapSubjectIDs(s__po_sid):
            (s, ((p, o), sid)) = s__po_sid
            return o, (sid, p)

        o__sid_p = s_po.join(entityIDs).map(mapSubjectIDs)
        p__oid_sid = o__sid_p.join(entityIDs).map(mapSubjectIDs)
        sid__pid_oid = p__oid_sid.join(relationIDs).map(mapSubjectIDs)

        spoMapped = sid__pid_oid.map(lambda x: (x[0], x[1][0], x[1][1]))
        assert isinstance(spoMapped, RDD)

        d1, d2, d3 = blockSizes

        def blockify(s, o, p):
            blockD1Index = int(s / d1)

        spoMapped.groupByKey().mapPartitions()


        return sid__pid_oid




os.environ['SPARK_HOME'] = "/Users/nilesh/IdeaProjects/spark-1.6.2-bin-hadoop2.6"
os.environ['PYSPARK_PYTHON'] = "python3"
os.environ['PYSPARK_DRIVER_PYTHON'] = "python3"
reader = RDFReader(SparkContext(master="local[4]", appName="test", sparkHome="/Users/nilesh/IdeaProjects/spark-1.6.2-bin-hadoop2.6"))
print(reader.tripleTensor("/Users/nilesh/IdeaProjects/elinker3/small-dataset.nt", 1).collect())