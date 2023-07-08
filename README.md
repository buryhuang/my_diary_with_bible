# My Dialog with Bible

## Initial Setup
Default setup of opensearch have watermark and read-only index. To remove it, run the following command:
```bash
docker-compose up -d
curl -XPUT -H "Content-Type: application/json" http://localhost:9200/_cluster/settings -d '{ "transient": { "cluster.routing.allocation.disk.threshold_enabled": false } }'
curl -XPUT -H "Content-Type: application/json" http://localhost:9200/_all/_settings -d '{"index.blocks.read_only_allow_delete": null}'
```

## Run the service
Assuming you have installed python3 locally, if not, please follow [this link](https://realpython.com/installing-python/).
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
flask run
```

## Initialize OpenSearch

Send a POST request to [the endpoint](app.py#L16)
```
curl --location --request POST 'http://127.0.0.1:5000/load'
```

If you don't have open api key set in your environment, stop flask, add the following to [.venv/bin/activate](.venv/bin/activate)
```
export OPENAI_API_KEY=YOUR_CHATGPT4_KEY
```

run `source .venv/bin/activate` to set the environment variable, start flask, and hit the load endpoint again. 

## Run the webapp
```bash
cd web
npm install
npm run build
npm run start
```
