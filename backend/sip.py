
from livekit import api

import logging
import os
import uuid
from datetime import datetime
from typing import Optional, Dict, Any


from dotenv import load_dotenv


logger = logging.getLogger("sip-setup")
load_dotenv()
# Load environment variables
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_URL = os.getenv("LIVEKIT_URL")



async def setup_sip_for_new_call(
    caller_phone_number: str,
    room_name: Optional[str] = None,
    participant_identity: Optional[str] = None,
) -> Dict[str, Any]:
        try:
            lkapi = api.LiveKitAPI(
                url=LIVEKIT_URL,
                api_key=LIVEKIT_API_KEY,
                api_secret=LIVEKIT_API_SECRET,
            )

            # Generate room name and participant identity if not provided
            if not room_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                room_name = f"call_{timestamp}_{uuid.uuid4().hex[:8]}"
            
            if not participant_identity:
                participant_identity = f"caller_{uuid.uuid4().hex[:8]}"

            logger.info(f"Setting up SIP for call from {caller_phone_number} to room {room_name}")

            # Step 1: Create Access Token
            token = (
            api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
            .with_identity(participant_identity)
            .with_grants(
                 api.VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_publish=True,
                    can_subscribe=True,
                    can_update_own_metadata=True,
                    )
                )
            .with_room_config(
            api.RoomConfiguration(
                agents=[
                    api.RoomAgentDispatch(
                        agent_name="assistant", metadata="test-metadata"
                            )
                        ],
                    ),
                )
            )
            token_jwt = token.to_jwt()
            logger.info(f"Created access token for participant {participant_identity}")




            #step 2: Create SIP Inbound Trunk if not exists


            in_trunks = await lkapi.sip.list_inbound_trunk(
            api.ListSIPInboundTrunkRequest()
            )
            print(f"{in_trunks}")

            if in_trunks.items:
                in_trunk = in_trunks.items[0]
                in_trunk_id = in_trunk.sip_trunk_id
                logger.info(f"Using existing inbound trunk: {in_trunk_id}")
            else:
                in_trunk = api.SIPInboundTrunkInfo(
                    name=f"inbound_call",
                    numbers=["+74992130459"],
                    allowed_addresses=[caller_phone_number],
                    krisp_enabled=True,  # если реально хотите Krips (Cloud)
                )
                request = api.CreateSIPInboundTrunkRequest(trunk=in_trunk)
                inbound_trunk = await lkapi.sip.create_sip_inbound_trunk(request)
                inbound_trunk_id = inbound_trunk.sip_trunk_id
                logger.info(f"Created inbound trunk: {inbound_trunk_id}")


            #step 3: Create SIP Dispatch Rule if not exists
            rules = await lkapi.sip.list_dispatch_rule(
            api.ListSIPDispatchRuleRequest()
            )
            print(f"{rules}")

            if rules.items:
                rule = rules.items[0]
                rule_id = rule.sip_dispatch_rule_id
                logger.info(f"Using existing dispatch rule: {rule_id}")
            else:
                rule = api.SIPDispatchRule(
                dispatch_rule_individual = api.SIPDispatchRuleIndividual(
                room_prefix = 'call_',
                )
                )

                request = api.CreateSIPDispatchRuleRequest(
                dispatch_rule = api.SIPDispatchRuleInfo(
                    rule = rule,
                    name = f"rule_{room_name}",
                    trunk_ids = [inbound_trunk_id],
                    room_config=api.RoomConfiguration(
                        agents=[api.RoomAgentDispatch(
                            agent_name="assistant",
                        )]
                    )
                )
                )

                dispatch = await lkapi.sip.create_sip_dispatch_rule(request)
                rule_id = dispatch.sip_dispatch_rule_id
                logger.info(f"Created dispatch rule: {rule_id}")
            #step 4: Create SIP Outbound Trunk if not exists

            out_trunks = await lkapi.sip.list_outbound_trunk(
            api.ListSIPOutboundTrunkRequest()
            )
            print(f"{out_trunks}")

            if out_trunks.items: 
                out_trunk = out_trunks.items[0]
                out_trunk_id = out_trunk.sip_trunk_id
                logger.info(f"Using existing outbound trunk: {out_trunk_id}")
            else:
                out_trunk = api.SIPOutboundTrunkInfo(
                    name=f"outbound_{room_name}",
                    # Add your SIP provider configuration here
                    # This is a placeholder - adjust based on your provider
                )
                request = api.CreateSIPOutboundTrunkRequest(trunk=out_trunk)
                outbound_trunk = await lkapi.sip.create_sip_outbound_trunk(request)
                outbound_trunk_id = outbound_trunk.sip_trunk_id
                logger.info(f"Created outbound trunk: {outbound_trunk_id}")

            print("setup complete")
            return {
                "token": token_jwt,
                "room_name": room_name,
                "participant_identity": participant_identity,
                "inbound_trunk_id": in_trunk_id,
                "outbound_trunk_id": out_trunk_id,
                "dispatch_rule_id": rule_id,
            }







        except Exception as e:
            logger.error(f"Error setting up SIP for new call: {e}")
            raise e
        finally:
            await lkapi.aclose()
        