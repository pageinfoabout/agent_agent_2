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
    
  
    rules = await lkapi.sip.list_dispatch_rule(
    api.ListSIPDispatchRuleRequest()
    )
    print(f"{rules}")

    trunks = await lkapi.sip.list_inbound_trunk(
    api.ListSIPInboundTrunkRequest()
    )
    print(f"{trunks}")

    if not rules.items:
        rule = api.SIPDispatchRule(
        dispatch_rule_individual = api.SIPDispatchRuleIndividual(
        room_prefix = 'call-',
        )
        )

        request = api.CreateSIPDispatchRuleRequest(
        dispatch_rule = api.SIPDispatchRuleInfo(
            rule = rule,
            name = 'My dispatch rule',
            trunk_ids = ["ST_xWa2NPEYShEZ"],
            room_config=api.RoomConfiguration(
                agents=[api.RoomAgentDispatch(
                    agent_name="assistant",
                )]
            )
        )
        )

        dispatch = await lkapi.sip.create_sip_dispatch_rule(request)

        
        print("created dispatch", dispatch)
        

        await lkapi.aclose()
            
    if not trunks.items:
        trunk = api.SIPInboundTrunkInfo(
        name = "My trunk",
        numbers = ["+74992130459"],
        allowed_numbers = {}
        krisp_enabled = True,
        )
        request = api.CreateSIPInboundTrunkRequest(
        trunk = trunk
        )
        trunk = await lkapi.sip.create_sip_inbound_trunk(request)
        print(f"Created trunk: {trunk.sip_trunk_id}")
        await lkapi.aclose()
    else:
        print("trunks and dispatch rule already exists")
        await lkapi.aclose()


  

 

asyncio.run(main())



    
