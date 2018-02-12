for x in ~/tensor-chess-data/all-pgns/*.pgn
do
echo "Starting $x"
name=$(basename $x)
sem -j 8 "python convert_pgns.py $x ~/tensor-chess-data/pgn-tfrecords/$name.tfrecord"
done
sem --wait
