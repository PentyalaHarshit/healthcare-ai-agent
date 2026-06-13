import requests
import json

BASE_URL = 'http://localhost:8000'

def run_test():
    print("=== STARTING MULTI-TURN AGENT TEST ===")

    # Step 1: Ask for the available hospitals
    print("\n[User]: can you give the list of hospital")
    response1 = requests.post(f"{BASE_URL}/chat", json={
        "message": "can you give the list of hospital"
    })
    result1 = response1.json()
    print(f"[Router Agent]: {result1.get('router_agent')}")
    print(f"[Healthcare Agent]: {result1.get('reply')}")

    # Step 2: Start chat session by describing symptoms
    print("\n[User]: I have chest pain and diabetes")
    response2 = requests.post(f"{BASE_URL}/chat", json={
        "message": "I have chest pain and diabetes"
    })
    result2 = response2.json()
    session_id = result2.get("session_id")
    print(f"[Healthcare Agent]: {result2.get('reply')}")
    print(f"(Session ID: {session_id})")

    # Step 3: Select a recommended hospital
    print("\n[User]: Texas Health Frisco")
    response3 = requests.post(f"{BASE_URL}/chat", json={
        "session_id": session_id,
        "message": "Texas Health Frisco"
    })
    result3 = response3.json()
    
    print("\n=== APPOINTMENT BOOKING SUMMARY ===")
    print(f"Hospital: {result3.get('hospital')}")
    print(f"Symptoms: {result3.get('symptoms')}")
    print(f"Risk Score: {result3.get('risk_score')}/100")
    print(f"Priority: {result3.get('priority')}")
    print(f"Recommended Specialty: {result3.get('recommended_specialty')}")
    print(f"XAI Reasons: {result3.get('xai_explanation')}")
    print(f"RAG Guidelines: {result3.get('rag_medical_context')}")
    print(f"Recommendation: {result3.get('recommendation')}")

    slots = result3.get('available_doctors', [])
    print(f"Available Slots: {len(slots)}")
    for slot in slots:
        print(f" - Slot ID: {slot['slot_id']} | Doctor: {slot['doctor_name']} | Date: {slot['date']} | Time: {slot['time']}")

    if slots:
        selected_slot = slots[0]['slot_id']
        print(f"\n[User]: Booking Slot ID {selected_slot}...")
        response4 = requests.post(f"{BASE_URL}/book", json={
            "slot_id": selected_slot
        })
        result4 = response4.json()
        print(f"[System]: Status: {result4.get('status')} | Message: {result4.get('message')}")
    else:
        print("\n[System]: No slots available to book.")
        if "suggested_hospitals" in result3:
            print(f"Suggested Alternate Locations: {result3['suggested_hospitals']}")

if __name__ == "__main__":
    run_test()
