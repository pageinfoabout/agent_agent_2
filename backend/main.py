import asyncio
from livekit import api
import os
from dotenv import load_dotenv

load_dotenv()

async def main():
    lkapi = api.LiveKitAPI(
        url=os.getenv("LIVEKIT_URL"),
        api_key=os.getenv("LIVEKIT_API_KEY"),
        api_secret=os.getenv("LIVEKIT_API_SECRET"),
    )

    # Get or create inbound trunk
    trunks = await lkapi.sip.list_inbound_trunk(api.ListSIPInboundTrunkRequest())
    print(f"Trunks: {trunks}")
    
    if trunks.items:
        in_trunk = trunks.items[0]
        in_trunk_id = in_trunk.sip_trunk_id
        print(f"Using existing inbound trunk: {in_trunk_id}")
    else:
        in_trunk = api.SIPInboundTrunkInfo(
            name="inbound_call",
            numbers=["+74992130459"],  # Added + for E.164 format
            krisp_enabled=True,
        )
        request = api.CreateSIPInboundTrunkRequest(trunk=in_trunk)
        inbound_trunk = await lkapi.sip.create_sip_inbound_trunk(request)
        in_trunk_id = inbound_trunk.sip_trunk_id
        print(f"Created inbound trunk: {in_trunk_id}")

    # Get or create dispatch rule
    rules = await lkapi.sip.list_dispatch_rule(api.ListSIPDispatchRuleRequest())
    print(f"Rules: {rules}")

    if not rules.items:
        rule = api.SIPDispatchRule(
            dispatch_rule_individual=api.SIPDispatchRuleIndividual(
                room_prefix="call-"
            )
        )
        
        request = api.CreateSIPDispatchRuleRequest(
            dispatch_rule=api.SIPDispatchRuleInfo(
                rule=rule,
                name="My dispatch rule",
                trunk_ids=[in_trunk_id],  # Fixed: use correct variable
                room_config=api.RoomConfiguration(
                    agents=[api.RoomAgentDispatch(agent_name="assistant")]
                )
            )
        )

        dispatch = await lkapi.sip.create_sip_dispatch_rule(request)
        print("Created dispatch:", dispatch)
    else:
        print("Dispatch rule already exists")

    await lkapi.aclose()

if __name__ == "__main__":
    asyncio.run(main())
