"""
Query Agent - Specialized in querying the database
"""
import sqlite3
import json
from typing import Dict, Any, List
from agents.base_agent import BaseAgent

class QueryAgent(BaseAgent):
    """Agent responsible for querying the database"""
    
    def __init__(self, db_path: str = "retail_banking.db", gemini_client=None):
        super().__init__("QueryAgent", "Database Query Specialist", gemini_client)
        self.db_path = db_path
    
    async def execute(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute database query based on natural language question
        """
        self.log(f"Processing query: {task}")
        
        # Use OpenAI to convert natural language to SQL
        sql_query = await self._generate_sql(task)
        self.log(f"Generated SQL: {sql_query}")
        
        # Execute query
        results = await self._execute_query(sql_query)
        
        return {
            "agent": self.name,
            "sql_query": sql_query,
            "results": results,
            "row_count": len(results),
            "status": "success"
        }
    
    async def _generate_sql(self, question: str) -> str:
        """Use Gemini to generate SQL from natural language"""
        
        schema_info = self._get_schema_info()
        
        system_prompt = """You are a SQL expert. Convert natural language questions to SQLite SQL queries.
        Only return the SQL query, nothing else. Do not include markdown code blocks.
        Use proper JOINs to maintain relationships between tables.
        Return only valid SQLite syntax."""
        
        user_prompt = f"""Database Schema:
{schema_info}

Question: {question}

Generate a SQL query to answer this question. Return ONLY the SQL query."""
        
        sql = await self._call_gemini(user_prompt, system_prompt, temperature=0.3)
        # Clean up the response (remove markdown if present)
        sql = sql.strip()
        if sql.startswith("```sql"):
            sql = sql[6:]
        if sql.startswith("```"):
            sql = sql[3:]
        if sql.endswith("```"):
            sql = sql[:-3]
        sql = sql.strip()
        
        return sql
    
    async def _execute_query(self, sql: str) -> List[Dict[str, Any]]:
        """Execute SQL query and return results"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute(sql)
            rows = cursor.fetchall()
            # Convert rows to dictionaries
            results = [dict(row) for row in rows]
            return results
        except Exception as e:
            self.log(f"Query error: {str(e)}")
            return [{"error": str(e)}]
        finally:
            conn.close()
    
    def _get_schema_info(self) -> str:
        """Get database schema information"""
        return """
        Tables and Relationships:
        
        1. customers (customer_id, first_name, last_name, email, phone, date_of_birth, created_at)
        2. addresses (address_id, customer_id, street_address, city, state, zip_code, country, address_type)
        3. branches (branch_id, branch_name, branch_code, address, city, phone, manager_name)
        4. departments (department_id, department_name, branch_id, budget)
        5. employees (employee_id, first_name, last_name, email, phone, hire_date, salary, department_id, manager_id)
        6. accounts (account_id, customer_id, account_number, account_type, balance, branch_id, opened_date, status)
        7. credit_cards (card_id, customer_id, account_id, card_number, card_type, expiry_date, credit_limit, current_balance)
        8. transactions (transaction_id, account_id, transaction_type, amount, transaction_date, description, employee_id)
        9. loans (loan_id, customer_id, account_id, loan_type, principal_amount, interest_rate, loan_date, maturity_date, status, employee_id)
        10. payments (payment_id, loan_id, payment_amount, payment_date, payment_type)
        11. categories (category_id, category_name, description, parent_category_id)
        12. suppliers (supplier_id, supplier_name, contact_name, email, phone, address)
        13. products (product_id, product_name, category_id, supplier_id, price, cost, description, sku)
        14. inventory (inventory_id, product_id, branch_id, quantity, reorder_level, last_restocked)
        15. orders (order_id, customer_id, order_date, total_amount, status, branch_id, employee_id)
        16. order_items (order_item_id, order_id, product_id, quantity, unit_price, subtotal)
        17. shipments (shipment_id, order_id, shipment_date, tracking_number, carrier, status, estimated_delivery)
        18. returns (return_id, order_id, product_id, return_date, return_reason, quantity, refund_amount, status)
        19. reviews (review_id, customer_id, product_id, order_id, rating, review_text, review_date)
        20. promotions (promotion_id, promotion_name, product_id, category_id, discount_percentage, start_date, end_date, status)
        
        Key Relationships:
        - customers -> addresses (one-to-many)
        - customers -> accounts (one-to-many)
        - customers -> orders (one-to-many)
        - customers -> loans (one-to-many)
        - accounts -> transactions (one-to-many)
        - accounts -> credit_cards (one-to-many)
        - loans -> payments (one-to-many)
        - orders -> order_items (one-to-many)
        - orders -> shipments (one-to-many)
        - products -> inventory (one-to-many across branches)
        - categories -> products (one-to-many)
        - suppliers -> products (one-to-many)
        - branches -> departments (one-to-many)
        - departments -> employees (one-to-many)
        """

