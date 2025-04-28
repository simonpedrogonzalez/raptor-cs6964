# run_streaming.py - corrected version
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr


import pickle
from io import BytesIO
import matplotlib.pyplot as plt

# Import your existing methods
from raptor_methods import *  # Adjust import as needed

# Functions to deserialize Kafka messages
def deserialize_tile(tile_bytes):
    if tile_bytes is not None:
        return pickle.loads(BytesIO(tile_bytes).getvalue())
    return None

def deserialize_polygons(poly_bytes):
    if poly_bytes is not None:
        return pickle.loads(BytesIO(poly_bytes).getvalue())
    return None

# Initialize Spark
spark = SparkSession.builder \
    .appName("RaptorStreaming") \
    .getOrCreate()

# Set log level to reduce noise
spark.sparkContext.setLogLevel("WARN")

# Read raw tiles from Kafka
tiles = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "raw-tiles") \
    .load() \
    .selectExpr("CAST(value AS BINARY) as tile_bytes", "timestamp")

# Read raw polygons from Kafka
polys = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "raw-polygons") \
    .load() \
    .selectExpr("CAST(value AS BINARY) as poly_bytes", "timestamp")

# Join tiles and polygons - FIX: Use explicit column references with col()
# Join tiles and polygons - using column objects
joined = tiles.join(
    polys,
    (tiles.timestamp >= polys.timestamp - expr("interval 1 minute")) & 
    (tiles.timestamp <= polys.timestamp + expr("interval 1 minute"))
)
# Process each micro-batch of joined data
def process_batch(df, epoch_id):
    # If batch is empty, return empty DataFrame
    if df.isEmpty():
        return
    
    # Define the schema for our results
    from pyspark.sql.types import StructType, StructField, StringType, TimestampType, FloatType
    schema = StructType([
        StructField("region_id", StringType(), False),
        StructField("timestamp", TimestampType(), False),
        StructField("stat_value", FloatType(), False)
    ])
    
    # Collect data for processing
    records = []
    for row in df.collect():
        try:
            # Deserialize tile and polygon data
            tile = deserialize_tile(row.tile_bytes)
            regions = deserialize_polygons(row.poly_bytes)
            
            if tile is not None and regions is not None:
                # Call your Raptor zonal stats function
                stats = raptor_zonal_stats(
                    tile['data'], 
                    regions['geometry']
                )
                
                # Create records for each region's stats
                for region_id, stat in stats.items():
                    records.append((
                        str(region_id), 
                        row.timestamp, 
                        float(stat)
                    ))
        except Exception as e:
            print(f"Error processing batch: {e}")
    
    # Create DataFrame from records
    if records:
        result_df = spark.createDataFrame(records, schema=schema)
        
        # Write results to Kafka
        result_df.selectExpr(
            "to_json(struct(*)) AS value"
        ).write \
            .format("kafka") \
            .option("kafka.bootstrap.servers", "localhost:9092") \
            .option("topic", "zonal-stats") \
            .save()

# Apply the batch processing function
query = joined.writeStream \
    .foreachBatch(process_batch) \
    .start()

# Wait for termination
query.awaitTermination()