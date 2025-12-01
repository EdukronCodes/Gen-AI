"""
Initialize Database with Sample Data
"""
import asyncio
from app.core.database import init_db, SessionLocal
from app.models.user import User, UserRole
from app.models.engineer import Engineer, EngineerSkill
from app.models.knowledge_base import KnowledgeBaseEntry
from app.core.security import get_password_hash


def create_sample_data():
    """Create sample users, engineers, and knowledge base entries"""
    db = SessionLocal()
    
    try:
        # Create admin user
        admin = User(
            email="admin@helpdesk.com",
            username="admin",
            hashed_password=get_password_hash("admin123"),
            full_name="Admin User",
            role=UserRole.ADMIN,
            is_active=True
        )
        db.add(admin)
        
        # Create engineer user
        engineer_user = User(
            email="engineer@helpdesk.com",
            username="engineer",
            hashed_password=get_password_hash("engineer123"),
            full_name="Engineer User",
            role=UserRole.ENGINEER,
            is_active=True
        )
        db.add(engineer_user)
        db.commit()
        db.refresh(engineer_user)
        
        # Create engineer
        engineer = Engineer(
            user_id=engineer_user.id,
            employee_id="ENG001",
            full_name="John Engineer",
            email="engineer@helpdesk.com",
            active_tickets_count=0,
            max_concurrent_tickets=10,
            is_available="true"
        )
        db.add(engineer)
        db.commit()
        db.refresh(engineer)
        
        # Add engineer skills
        skills = [
            EngineerSkill(engineer_id=engineer.id, skill_name="network", proficiency_level=8),
            EngineerSkill(engineer_id=engineer.id, skill_name="server", proficiency_level=7),
            EngineerSkill(engineer_id=engineer.id, skill_name="database", proficiency_level=6),
        ]
        for skill in skills:
            db.add(skill)
        
        # Create knowledge base entries
        kb_entries = [
            KnowledgeBaseEntry(
                title="Server Restart Script",
                content="How to restart a server when it becomes unresponsive",
                category="server",
                sub_category="restart",
                resolution_steps=["1. Check server status", "2. Restart service", "3. Verify"],
                resolution_script="systemctl restart myservice",
                success_rate=95
            ),
            KnowledgeBaseEntry(
                title="Database Connection Reset",
                content="Reset database connections when connection pool is exhausted",
                category="database",
                sub_category="connection",
                resolution_steps=["1. Check connection pool", "2. Reset connections", "3. Verify"],
                resolution_script="kubectl rollout restart deployment/database",
                success_rate=85
            ),
        ]
        
        for entry in kb_entries:
            db.add(entry)
        
        db.commit()
        print("✅ Sample data created successfully")
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error creating sample data: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(init_db())
    create_sample_data()


