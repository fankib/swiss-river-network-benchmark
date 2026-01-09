#python swissrivernetwork/benchmark/ray_tune_embedding.py -m concatenation_embedding -g swiss-1990 -n 200
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m concatenation_embedding_output -g swiss-1990 -n 200
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m embedding_gate_memory -g swiss-1990 -n 200
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m embedding_gate_hidden -g swiss-1990 -n 200
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m interpolation_embedding -g swiss-1990 -n 200

#python swissrivernetwork/benchmark/ray_tune_embedding.py -m concatenation_embedding -g swiss-2010 -n 200
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m concatenation_embedding_output -g swiss-2010 -n 200
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m embedding_gate_memory -g swiss-2010 -n 200
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m embedding_gate_hidden -g swiss-2010 -n 200
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m interpolation_embedding -g swiss-2010 -n 200

# concatenation vs interpolation (using scheduler_long())
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m concatenation_embedding -g swiss-1990 -n 100
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m interpolation_embedding -g swiss-1990 -n 100
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m embedding_gate_memory -g swiss-1990 -n 100 # optional


# Use small embeddings (1,2,5)
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m concatenation_embedding -g swiss-1990 -n 50
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m concatenation_embedding_output -g swiss-1990 -n 50
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m embedding_gate_memory -g swiss-1990 -n 50
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m embedding_gate_hidden -g swiss-1990 -n 50
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m interpolation_embedding -g swiss-1990 -n 50

#python swissrivernetwork/benchmark/ray_tune_embedding.py -m concatenation_embedding -g swiss-2010 -n 50
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m concatenation_embedding_output -g swiss-2010 -n 50
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m embedding_gate_memory -g swiss-2010 -n 50
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m embedding_gate_hidden -g swiss-2010 -n 50
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m interpolation_embedding -g swiss-2010 -n 50


# use static embeddings: (epochs, emb_size=3)
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m concatenation_embedding -g swiss-1990 -n 100 --static_embedding
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m concatenation_embedding_output -g swiss-1990 -n 100 --static_embedding
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m embedding_gate_memory -g swiss-1990 -n 100 --static_embedding
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m embedding_gate_hidden -g swiss-1990 -n 100 --static_embedding
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m interpolation_embedding -g swiss-1990 -n 100 --static_embedding

#python swissrivernetwork/benchmark/ray_tune_embedding.py -m concatenation_embedding -g swiss-2010 -n 100 --static_embedding
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m concatenation_embedding_output -g swiss-2010 -n 100 --static_embedding
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m embedding_gate_memory -g swiss-2010 -n 100 --static_embedding
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m embedding_gate_hidden -g swiss-2010 -n 100 --static_embedding
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m interpolation_embedding -g swiss-2010 -n 100 --static_embedding

# Run static embeddings with shuffled coordinates
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m concatenation_embedding -g swiss-1990 -n 100 --static_embedding --shuffle_embedding
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m concatenation_embedding_output -g swiss-1990 -n 100 --static_embedding --shuffle_embedding
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m embedding_gate_memory -g swiss-1990 -n 100 --static_embedding --shuffle_embedding
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m embedding_gate_hidden -g swiss-1990 -n 100 --static_embedding --shuffle_embedding
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m interpolation_embedding -g swiss-1990 -n 100 --static_embedding --shuffle_embedding

#python swissrivernetwork/benchmark/ray_tune_embedding.py -m concatenation_embedding -g swiss-2010 -n 100 --static_embedding --shuffle_embedding
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m concatenation_embedding_output -g swiss-2010 -n 100 --static_embedding --shuffle_embedding
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m embedding_gate_memory -g swiss-2010 -n 100 --static_embedding --shuffle_embedding
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m embedding_gate_hidden -g swiss-2010 -n 100 --static_embedding --shuffle_embedding
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m interpolation_embedding -g swiss-2010 -n 100 --static_embedding --shuffle_embedding

# Run static embeddings, picked at random
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m concatenation_embedding -g swiss-1990 -n 100 --static_embedding --random_embedding
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m embedding_gate_memory -g swiss-1990 -n 100 --static_embedding --random_embedding
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m interpolation_embedding -g swiss-1990 -n 100 --static_embedding --random_embedding

#python swissrivernetwork/benchmark/ray_tune_embedding.py -m concatenation_embedding -g swiss-2010 -n 100 --static_embedding --random_embedding
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m embedding_gate_memory -g swiss-2010 -n 100 --static_embedding --random_embedding
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m interpolation_embedding -g swiss-2010 -n 100 --static_embedding --random_embedding

# Run static embeddings, picked at random
python swissrivernetwork/benchmark/ray_tune_embedding.py -m concatenation_embedding -g swiss-1990 -n 100 --static_embedding --one_hot_embedding
python swissrivernetwork/benchmark/ray_tune_embedding.py -m embedding_gate_memory -g swiss-1990 -n 100 --static_embedding --one_hot_embedding
python swissrivernetwork/benchmark/ray_tune_embedding.py -m interpolation_embedding -g swiss-1990 -n 100 --static_embedding --one_hot_embedding

python swissrivernetwork/benchmark/ray_tune_embedding.py -m concatenation_embedding -g swiss-2010 -n 100 --static_embedding --one_hot_embedding
python swissrivernetwork/benchmark/ray_tune_embedding.py -m embedding_gate_memory -g swiss-2010 -n 100 --static_embedding --one_hot_embedding
python swissrivernetwork/benchmark/ray_tune_embedding.py -m interpolation_embedding -g swiss-2010 -n 100 --static_embedding --one_hot_embedding

shutdown


# Run Vanilla LSTM
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m vanilla -g swiss-1990 -n 100
#python swissrivernetwork/benchmark/ray_tune_embedding.py -m vanilla -g swiss-2010 -n 100

