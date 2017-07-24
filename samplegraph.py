#!/bin/python

import codecs
import random
import itertools
import numpy as np
from collections import Counter
import multiprocessing as mp
from pyspark import SparkContext, SparkConf


langs = ['en', 'de', 'fr', 'nl', 'it', 'es', 'ro', 'pl', 'ar', 'fa']

with codecs.open("/data/yago/entities.txt", encoding='utf-8') as e:
	with codecs.open("/data/yago/relations.txt", encoding='utf-8') as r:
		with codecs.open("/data/yago/all.txt", encoding='utf-8') as t:
			triples = (tuple(i.split()) for i in t.readlines())
			triples2 = [i for i in triples if not any([(lang in i[0] or lang in i[1]) for lang in langs])]
			relations = set([i.strip() for i in r.readlines()])
			counter = Counter(itertools.chain(*[[triple[0], triple[2]] for triple in triples2]))

selectEntities = set([i[0] for i in counter.most_common(1000)])
print "Initial selectEntities", len(selectEntities)
entities = set(counter)
selectTriples = set()

entities.difference_update(selectEntities)

# Generating subgraph

count = 0
new = set()
while len(selectEntities) < 100000:
	count += 1
	print count
	selectEntity = set(random.sample(entities, int(1000)))
	print len(selectEntity)
	entities.difference_update(selectEntity)
	selectEntity.update(new)
	print len(selectEntity)
	print "Getting newtriples"
	newTriples = [triple for triple in triples2 if (triple[0] in selectEntity or triple[1] in selectEntity)]
	print len(newTriples)
	print "Getting newentities"
	newEntities = set(itertools.chain(*[[triple[0], triple[2]] for triple in newTriples]))
	print len(newEntities)
	print "just new?"
	new = newEntities.difference(selectEntities)
	print len(new)
	print "updating"
	selectEntities.update(newEntities)
	selectTriples.update(newTriples)
	print "Sizes:", len(selectEntities), len(selectTriples)



# Pruning
entityCounts = Counter(itertools.chain(*[[triple[0], triple[2]] for triple in selectTriples]))
selectEntities = set([i[0] for i in entityCounts.most_common(100000)])
selectTriples = set([triple for triple in selectTriples if (triple[0] in selectEntities or triple[1] in selectEntities)])
print "Final sizes:", len(selectEntities), len(selectTriples)

with codecs.open("/data/yago/subgraph/entity2id.txt", "w", encoding='utf-8') as entity2id:
	for i, entity in enumerate(selectEntities):
		entity2id.write("%s\t%s\n" % (entity, i))

with codecs.open("/data/yago/subgraph/triples.txt", "w", encoding='utf-8') as triples:
	for i, triple in enumerate(selectTriples):
		triples.write("%s\t%s\t%s\n" % triple)

conf = SparkConf().setAppName("yago-preprocess").setMaster("local[%d]" % mp.cpu_count())
sc = SparkContext(conf=conf)

sebc = sc.broadcast(selectEntities)

for auxFile in ["/data/yago/aux/yagoLabels.tsv",
			"/data/yago/aux/yagoTransitiveType.tsv",
			"/data/yago/aux/yagoLiteralFacts.tsv",
			"/data/yago/aux/yagoDateFacts.tsv",]:
	sc.textFile(auxFile) \
	.map(lambda x: x.split()) \
	.flatMap(lambda x: [x[1:]] if x[1] in sebc.value else []) \
	.map(lambda x: "\t".join(x)) \
	.saveAsTextFile("/data/yago/aux/aux-" + auxFile.split("/")[-1])
