# Define prompts
MIXED_PROMPT = """You are a helpful assistant. Your task is to generate realistic customer feedback that specifically demonstrates Mixed sentiment for electrical devices (circuit breakers, transformers, smart meters, inverters, solar panels, and power strips). 

For Mixed sentiment, create feedback that (in 4-5 lines):
- Contains both positive and negative aspects about the same device
- Balances pros and cons
- Shows uncertainty or conflicting experiences
- Demonstrates realistic technical observations

Example: "I installed the 7.5kW solar inverter three months ago. The efficiency ratings are impressive, consistently hitting 97-98% during peak hours, and the LCD display is clear and informative. However, the WiFi monitoring system has been unreliable, disconnecting at least twice a week requiring manual resets. The mobile app also needs improvement, though the basic monitoring features work when connected. Despite these connectivity issues, the core power conversion function works perfectly."

Please generate one Mixed sentiment feedback example:"""

NEUTRAL_PROMPT = """You are a helpful assistant. Your task is to generate realistic customer feedback that specifically demonstrates Neutral sentiment for electrical devices (circuit breakers, transformers, smart meters, inverters, solar panels, and power strips).

For Neutral sentiment, create feedback that (in 4-5 lines):
- Is purely factual and objective
- Avoids any emotional language or personal opinions
- Focuses on technical specifications or basic observations
- Uses matter-of-fact descriptions of functionality
- States observations about installation, operation, or maintenance
- Describes standard performance without evaluative judgments
- Focuses on measurable or observable characteristics

Example: "The 100A circuit breaker was installed in the main distribution panel on June 15th. The unit operates at 240V and maintains a steady temperature of 35°C under normal load conditions. Monthly testing shows the trip mechanism responds within 50 milliseconds when exceeding rated current. The installation required standard mounting hardware and took approximately 45 minutes to complete. The breaker logs show it has tripped three times during power surges since installation."

Please generate one Neutral sentiment feedback example:"""
