"""
Rewards/Loyalty Agent - Handles loyalty programs and rewards
"""
from typing import Dict, Any
from sqlalchemy.orm import Session
from .base_agent import BaseAgent
from database.models import Passenger, Booking


class RewardsAgent(BaseAgent):
    """Agent specialized in loyalty programs and rewards"""
    
    def __init__(self, db: Session):
        super().__init__(
            name="Rewards & Loyalty Agent",
            description="an expert on loyalty programs, rewards points, membership tiers, and benefits",
            db=db
        )
    
    def get_knowledge_documents(self) -> list[str]:
        return [
            "Loyalty program tiers: Bronze, Silver, Gold, Platinum.",
            "Points are earned based on flight distance and fare class.",
            "Bronze tier: 1x points, Silver: 1.25x points, Gold: 1.5x points, Platinum: 2x points.",
            "Points can be redeemed for flights, upgrades, and partner services.",
            "Status benefits include priority boarding, lounge access, and extra baggage allowance.",
            "Points expire after 24 months of inactivity.",
        ]
    
    def get_loyalty_info(self, passenger_email: str) -> Dict[str, Any]:
        """Get loyalty program information for a passenger"""
        passenger = self.db.query(Passenger).filter(
            Passenger.email == passenger_email
        ).first()
        
        if not passenger:
            return {"success": False, "error": "Passenger not found"}
        
        # Calculate points from bookings
        bookings = self.db.query(Booking).filter(
            Booking.passenger_id == passenger.id
        ).all()
        
        total_spent = sum(booking.price for booking in bookings)
        
        # Tier benefits
        tier_benefits = {
            "bronze": {"points_multiplier": 1.0, "lounge_access": False, "priority_boarding": False},
            "silver": {"points_multiplier": 1.25, "lounge_access": False, "priority_boarding": True},
            "gold": {"points_multiplier": 1.5, "lounge_access": True, "priority_boarding": True},
            "platinum": {"points_multiplier": 2.0, "lounge_access": True, "priority_boarding": True},
        }
        
        benefits = tier_benefits.get(passenger.membership_tier.lower(), tier_benefits["bronze"])
        
        return {
            "success": True,
            "passenger_name": f"{passenger.first_name} {passenger.last_name}",
            "membership_tier": passenger.membership_tier,
            "loyalty_points": passenger.loyalty_points,
            "total_bookings": len(bookings),
            "total_spent": total_spent,
            "benefits": benefits,
            "points_value": f"${passenger.loyalty_points * 0.01:.2f}"  # 1 point = $0.01
        }
    
    def redeem_points(self, passenger_email: str, points: int) -> Dict[str, Any]:
        """Redeem loyalty points"""
        passenger = self.db.query(Passenger).filter(
            Passenger.email == passenger_email
        ).first()
        
        if not passenger:
            return {"success": False, "error": "Passenger not found"}
        
        if passenger.loyalty_points < points:
            return {
                "success": False,
                "error": f"Insufficient points. You have {passenger.loyalty_points} points."
            }
        
        passenger.loyalty_points -= points
        self.db.commit()
        
        return {
            "success": True,
            "points_redeemed": points,
            "remaining_points": passenger.loyalty_points,
            "value": f"${points * 0.01:.2f}"
        }
    
    def process(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process rewards/loyalty request"""
        context = context or {}
        user_lower = user_input.lower()
        
        passenger_email = context.get("passenger_email")
        
        if passenger_email:
            if "redeem" in user_lower:
                points = context.get("points", 0)
                result = self.redeem_points(passenger_email, points)
                if result["success"]:
                    response_text = f"✅ Points redeemed successfully!\n\n"
                    response_text += f"Points Redeemed: {result['points_redeemed']}\n"
                    response_text += f"Value: {result['value']}\n"
                    response_text += f"Remaining Points: {result['remaining_points']}\n"
                else:
                    response_text = result.get("error", "Could not redeem points")
            else:
                # Get loyalty info
                result = self.get_loyalty_info(passenger_email)
                if result["success"]:
                    response_text = f"Loyalty Program Information:\n\n"
                    response_text += f"Member: {result['passenger_name']}\n"
                    response_text += f"Tier: {result['membership_tier'].upper()}\n"
                    response_text += f"Points: {result['loyalty_points']} ({result['points_value']} value)\n"
                    response_text += f"Total Bookings: {result['total_bookings']}\n"
                    response_text += f"Total Spent: ${result['total_spent']:.2f}\n\n"
                    response_text += f"Benefits:\n"
                    response_text += f"  Points Multiplier: {result['benefits']['points_multiplier']}x\n"
                    response_text += f"  Lounge Access: {'Yes' if result['benefits']['lounge_access'] else 'No'}\n"
                    response_text += f"  Priority Boarding: {'Yes' if result['benefits']['priority_boarding'] else 'No'}\n"
                else:
                    response_text = result.get("error", "Could not retrieve loyalty information")
        else:
            # General rewards information
            kb_response = self.query_knowledge_base(user_input)
            if kb_response:
                response_text = kb_response
            else:
                response_text = self.generate_response(
                    user_input,
                    "You are a loyalty program specialist. Provide information about membership tiers, points, benefits, and redemption options."
                )
        
        return {
            "agent": self.name,
            "response": response_text,
            "success": True
        }


