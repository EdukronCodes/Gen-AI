#!/usr/bin/env python3

import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the server directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'server'))

from app import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.getenv('NODE_ENV') == 'development')