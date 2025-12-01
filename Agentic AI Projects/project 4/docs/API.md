# API Documentation

## Base URL
```
http://localhost:8000/api/v1
```

## Authentication
Most endpoints require JWT authentication. Include the token in the Authorization header:
```
Authorization: Bearer <token>
```

## Endpoints

### Authentication

#### POST /auth/token
Login and get access token

**Request:**
```json
{
  "username": "admin",
  "password": "admin123"
}
```

**Response:**
```json
{
  "access_token": "eyJ...",
  "token_type": "bearer"
}
```

#### POST /auth/register
Register new user

**Request:**
```json
{
  "email": "user@example.com",
  "username": "user",
  "password": "password123",
  "full_name": "User Name"
}
```

### Tickets

#### POST /tickets/
Create a new ticket

**Request:**
```json
{
  "title": "Server Down",
  "description": "The production server is not responding",
  "channel": "web",
  "impact": "high"
}
```

**Response:**
```json
{
  "id": 1,
  "ticket_number": "TKT-20240101-ABC12345",
  "title": "Server Down",
  "status": "created",
  "priority": "P1"
}
```

#### GET /tickets/{ticket_id}
Get ticket by ID

#### GET /tickets/
List all tickets

### Orchestrator

#### POST /orchestrator/process
Process workflow stage

**Request:**
```json
{
  "stage": "intake",
  "ticket_id": 1,
  "data": {}
}
```

### Health

#### GET /health/
Health check

#### GET /health/detailed
Detailed health check with dependencies


