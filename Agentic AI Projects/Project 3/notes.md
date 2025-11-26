#### Multi Agent Airlines System

7 Agents
filght search agent : find and searches for avaliable flights
flight booking agent : create and manages resevations
customer service agent : handles general inquiries and support 
bagge informaiton agent : provides baggage policies and tracking
check in agent : process flight check in infromaiton
flight status aget : provides realtime flight status
rewards & loyalth agnt : manages all of you loyalty programs

==========================================================

LLamaIndex 
FastAPI
Postgress SQL (SQLlite3 )
        Database Initialization
        Data Models
        Insert data
        
===========================================================

## 🏗️ Architecture

```

```
┌─────────────────┐
│       UI   │
└────────┬────────┘
┌─────────────────┐
│   FastAPI App   │
└────────┬────────┘
         │
┌────────▼────────┐
│  Orchestrator (LLAMAINDEX)  │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐  ┌──▼───┐  ... (7 Agents)
│ Agent │  │Agent │
└───┬───┘  └──┬───┘
    │         │
┌───▼─────────▼───┐
│  PostgreSQL DB  │
└─────────────────┘

#### Prerequisites :

    python 3.11
    postgress sql 12+
    open api key
    docker
    azure account (portal azure.com)

##### fast api app : uvicorn main:app --reload -ost 0.0.0.0 --port 8000

loclahost:8000/ui : Front end 
localhost:8000/api/query
    {
        "query": "find flights from newyork to london"
        "response" : Table or json file with all detials
    }
localhost:8000/api/flights//departure_city=newyork&arrival_city=London

================================================



Docker deoployment


docker build -t airlinees-multi-agent :latest .


docker run -d -p 8000:8000

### deployment in 3 different ways


    ACI (azure container instance) ### responbility of devops
    Azure APP Services ## we can deploye from azure devops of github 
    AKS (Azure kubenetes services) ### reponsibliity of devops