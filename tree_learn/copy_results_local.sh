#! /usr/bin/env bash

# droplet_id = "493749359"
# droplet_ip = "162.243.45.138"
# volume_id = "ce287d73-29bc-11f0-b8da-0a58ac140531"

scp -r root@162.243.45.138:/mnt/tlvol/pipeline/ /media/penguaman/code/ActualCode/IaC/ml_ops_tree_learn/remote_data/WRRR/

# scp -r /mnt/tlvol/pipeline/ wischmcj@192.168.0.105:/media/penguaman/code/ActualCode/external/TreeLearn/remote_data/collective
