# My Dialog with Bible

## Initial Setup
Default setup of opensearch have watermark and read-only index. To remove it, run the following command:
```bash
docker-compose up -d
curl -XPUT -H "Content-Type: application/json" http://localhost:9200/_cluster/settings -d '{ "transient": { "cluster.routing.allocation.disk.threshold_enabled": false } }'
curl -XPUT -H "Content-Type: application/json" http://localhost:9200/_all/_settings -d '{"index.blocks.read_only_allow_delete": null}'
```

## Run the service
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install flask flask_cors llama_index
flask run
```

## Run the webapp
```bash
cd web
npm install
npm run build
npm run start
```
