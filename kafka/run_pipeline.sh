#!/bin/bash

# Kafka path
KAFKA_BIN="/opt/homebrew/bin"

# Start Kafka (which should handle its own coordination in KRaft mode)
$KAFKA_BIN/kafka-server-start /opt/homebrew/etc/kafka/server.properties &
KAFKA_PID=$!
echo "Started Kafka (PID: $KAFKA_PID)"

# Wait for Kafka to start
sleep 15

# Check if topics exist, create them if they don't
TOPICS=$($KAFKA_BIN/kafka-topics --list --bootstrap-server localhost:9092)

if [[ ! $TOPICS == *"raw-tiles"* ]]; then
    echo "Creating topic: raw-tiles"
    $KAFKA_BIN/kafka-topics --create --topic raw-tiles \
      --bootstrap-server localhost:9092 \
      --partitions 4 --replication-factor 1
fi

if [[ ! $TOPICS == *"raw-polygons"* ]]; then
    echo "Creating topic: raw-polygons"
    $KAFKA_BIN/kafka-topics --create --topic raw-polygons \
      --bootstrap-server localhost:9092 \
      --partitions 4 --replication-factor 1
fi

if [[ ! $TOPICS == *"zonal-stats"* ]]; then
    echo "Creating topic: zonal-stats"
    $KAFKA_BIN/kafka-topics --create --topic zonal-stats \
      --bootstrap-server localhost:9092 \
      --partitions 4 --replication-factor 1
fi

# Start the consumer in the background
echo "Starting consumer..."
python consume_stats.py &
CONSUMER_PID=$!

# Start Spark streaming job
echo "Starting Spark streaming job..."
spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0 \
  --driver-memory 4g \
  run_streaming.py &
SPARK_PID=$!

# Wait for Spark job to start
sleep 10

# Start the producers
echo "Starting polygon producer..."
python produce_polygons.py &
POLY_PRODUCER_PID=$!

echo "Starting tile producer..."
python produce_tiles.py &
TILE_PRODUCER_PID=$!

# Function to handle script termination
cleanup() {
    echo "Shutting down..."
    kill $TILE_PRODUCER_PID $POLY_PRODUCER_PID $SPARK_PID $CONSUMER_PID $KAFKA_PID
    $KAFKA_BIN/kafka-server-stop
    wait
}

# Set up trap to catch termination signal
trap cleanup EXIT

# Wait for input to terminate
echo "Pipeline running. Press any key to terminate..."
read -n 1

# Cleanup is called automatically by the trap