PORT_ON_HOST=$1
HOST_PORT=$2
HOST_IP=$3
ssh -R $PORT_ON_HOST:$(sudo service ssh start | sudo netstat -pan | grep sshd | grep LISTEN -m 1 | awk '{print $4}') -tt -p $HOST_PORT -o "StrictHostKeyChecking no" $HOST_IP