#!/bin/bash

# Numero totale di combinazioni
TOTAL=280720440
# Dimensione di ciascun chunk (ad esempio 40 milioni)
CHUNK=40000000
# Numero di fault simultanei (solo per nome file, opzionale)
N_FAULTS=4

START=0

while [ $START -lt $TOTAL ]
do
    END=$((START + CHUNK))
    if [ $END -gt $TOTAL ]; then
        END=$TOTAL
    fi
    echo "Lancio chunk: $START - $END"
    # Lancia il job in background con nohup e output separato
    nohup python main_online_minimal.py $START $END > log_${START}_${END}_N${N_FAULTS}.txt 2>&1 &
    START=$END
done
