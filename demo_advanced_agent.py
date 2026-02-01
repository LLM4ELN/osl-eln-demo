"""Demo script for the OoldAgent - creates entities from natural language."""

import time
import uuid

from opensemantic.v1 import OswBaseModel
from opensemantic.core.v1 import Entity

from oold_agent import OoldAgent
from osw.core import OSW
from osl_init import get_osl_client


# Main execution
start_time = time.time()

# Create agent instance
agent = OoldAgent()

# Create entity from description
result = agent.invoke(
    prompt=(
        "A laboratory process to document an experiment "
        "created by Dr. Jane Doe, Example Lab Corp., "
        "starting at 05.03.2025 and ending at 06.03.2025, "
        "status in finished."
    ),
)

# Print results
print("\n\n=== Created / Looked up entities ===")
entities = agent.get_entities()
for i, e in entities.items():
    e: OswBaseModel
    print(f"#### {i} ({e.name}) ####")
    print(e.json(indent=2, exclude_none=True))

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nElapsed time: {elapsed_time:.2f} seconds")

# Generate a short random id prefix
id_prefix = uuid.uuid4().hex[:6]

# Prefix all entity names with the id_prefix to avoid name collisions
for i, e in entities.items():
    e: Entity
    if e.name is not None:
        e.name = f"{id_prefix}_{e.name}"
    if e.label is not None:
        for lb in e.label:
            lb.text = f"{id_prefix} {lb.text}"

STORE = False
# Uncomment to store entities in OSL
# STORE = True
if STORE:
    print("\n\n=== Storing entities in OSL ===")
    osl_client = get_osl_client()
    osl_client.store_entity(OSW.StoreEntityParam(
        entities=list(entities.values()),
        overwrite=True,
        change_id="demo_advanced_agent-0002",
    ))
