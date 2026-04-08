# Databricks notebook source
# MAGIC %md
# MAGIC # Lakebase Memory Integration Test
# MAGIC Standalone test for the DatabricksStore-based session history and long-term memory.
# MAGIC
# MAGIC **Run this notebook on any Databricks cluster** — no pipeline needed.
# MAGIC It validates:
# MAGIC 1. DatabricksStore connection to Lakebase Autoscaling
# MAGIC 2. Short-term conversation history (save/load per thread_id)
# MAGIC 3. Long-term user memory (save/recall with semantic search)
# MAGIC 4. SessionStore class integration (full config-driven flow)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Install dependencies

# COMMAND ----------

# Install databricks-langchain with memory extras (DatabricksStore)
import subprocess
subprocess.check_call(["pip", "install", "-U", "-q", "databricks-langchain[memory]", "databricks-sdk"])
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configure — edit these values for your workspace

# COMMAND ----------

# Lakebase Autoscaling project — create one via CLI if it doesn't exist:
#   databricks lakebase projects create --name agentops-sessions
LAKEBASE_PROJECT = "agentops-sessions"
LAKEBASE_BRANCH = "production"  # matches config.yaml

# Test identifiers
TEST_THREAD_ID = "test-thread-001"
TEST_USER_ID = "test-user@example.com"

# Full config dict (mirrors config.yaml structure)
TEST_CONFIG = {
    "catalog": "classic_stable_cykcbe_catalog",
    "schema": "agentops",
    "session_history": {
        "enabled": True,
        "unity_catalog": {"enabled": False},  # Skip UC for this test
        "lakebase": {
            "enabled": True,
            "project": LAKEBASE_PROJECT,
            "branch": LAKEBASE_BRANCH,
        },
    },
    "long_term_memory": {
        "enabled": True,
    },
}

print(f"Lakebase project: {LAKEBASE_PROJECT}")
print(f"Lakebase branch:  {LAKEBASE_BRANCH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Verify Lakebase project exists

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
print(f"Workspace: {w.config.host}")
print(f"User: {w.current_user.me().user_name}")

# List Lakebase projects to verify ours exists
try:
    projects = w.api_client.do("GET", "/api/2.0/lakebase/projects")
    project_names = [p.get("name", "") for p in projects.get("projects", [])]
    if LAKEBASE_PROJECT in project_names:
        print(f"✓ Lakebase project '{LAKEBASE_PROJECT}' exists")
    else:
        print(f"✗ Project '{LAKEBASE_PROJECT}' not found. Available: {project_names}")
        print(f"  Create it: databricks lakebase projects create --name {LAKEBASE_PROJECT}")
except Exception as e:
    print(f"Could not list projects (may need different API path): {e}")
    print("Continuing anyway — DatabricksStore.setup() will fail clearly if project is missing.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Test DatabricksStore directly

# COMMAND ----------

from databricks_langchain import DatabricksStore

store = DatabricksStore(
    project=LAKEBASE_PROJECT,
    branch=LAKEBASE_BRANCH,
    workspace_client=w,
)

store.setup()
print("✓ DatabricksStore.setup() succeeded — Lakebase tables created")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Test short-term memory — conversation history per thread

# COMMAND ----------

import uuid
from datetime import datetime, timezone

thread_id = f"test-thread-{uuid.uuid4().hex[:8]}"

# Save a conversation
conversation = {
    "messages": [
        {"role": "user", "content": "What is Delta Lake?"},
        {"role": "assistant", "content": "Delta Lake is an open-source storage layer..."},
        {"role": "user", "content": "How does it handle ACID transactions?"},
        {"role": "assistant", "content": "Delta Lake provides ACID guarantees through..."},
    ],
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "turn_count": 2,
}

store.put(("conversations",), thread_id, conversation)
print(f"✓ Saved conversation with {len(conversation['messages'])} messages to thread {thread_id}")

# Read it back
item = store.get(("conversations",), thread_id)
assert item is not None, "Failed to retrieve conversation"
assert len(item.value["messages"]) == 4, f"Expected 4 messages, got {len(item.value['messages'])}"
print(f"✓ Retrieved conversation: {len(item.value['messages'])} messages, turn_count={item.value['turn_count']}")

# Verify message content round-trips correctly
assert item.value["messages"][0]["content"] == "What is Delta Lake?"
assert item.value["messages"][1]["role"] == "assistant"
print("✓ Message content integrity verified")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Test long-term memory — user facts with semantic search

# COMMAND ----------

user_id = f"test-user-{uuid.uuid4().hex[:8]}"

# Save user facts
store.put(("users", user_id), "name", {"content": "Alex Chen"})
store.put(("users", user_id), "role", {"content": "Data Engineer at Acme Corp"})
store.put(("users", user_id), "preferences", {"content": "Prefers Python over Scala, uses PySpark daily"})
store.put(("users", user_id), "budget", {"content": "Enterprise license, 500 DBU/month budget"})
store.put(("users", user_id), "past_topics", {"content": "Previously asked about Delta Lake optimization and Unity Catalog setup"})

print(f"✓ Saved 5 facts for user {user_id}")

# Semantic search — find relevant memories
results = store.search(("users", user_id), query="programming language preference", limit=3)
print(f"\n--- Semantic search: 'programming language preference' ---")
for r in results:
    print(f"  [{r.key}]: {r.value['content']}")
assert len(results) > 0, "Semantic search returned no results"
print(f"✓ Semantic search returned {len(results)} results")

# Another query
results2 = store.search(("users", user_id), query="what has the user asked about before", limit=3)
print(f"\n--- Semantic search: 'what has the user asked about before' ---")
for r in results2:
    print(f"  [{r.key}]: {r.value['content']}")
print(f"✓ Second search returned {len(results2)} results")

# Direct lookup
item = store.get(("users", user_id), "name")
assert item is not None and item.value["content"] == "Alex Chen"
print(f"\n✓ Direct lookup: name = {item.value['content']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Test SessionStore class (full config-driven flow)

# COMMAND ----------

import sys, os, subprocess

# Install the framework wheel from bundle to get the updated SessionStore
_bundle_root = "/Workspace/Users/narendra.merla@databricks.com/.bundle/agentops/dev/files"
subprocess.check_call(["pip", "install", "--force-reinstall", "--no-deps", "-q", _bundle_root])

# Also add bundle src to path (for direct imports)
_project_src = os.path.join(_bundle_root, "src")
if _project_src not in sys.path:
    sys.path.insert(0, _project_src)

from framework.session.session_store import SessionStore

session_store = SessionStore(TEST_CONFIG)
print(f"✓ SessionStore initialized: enabled={session_store.enabled}, lakebase={session_store.lakebase_enabled}, memory={session_store.memory_enabled}")

# COMMAND ----------

# Test: save_full_session + get_history
test_thread = f"session-test-{uuid.uuid4().hex[:8]}"

# Simulate messages (as dicts, like ChatAgentMessage would be serialized)
messages = [
    {"role": "user", "content": "Tell me about Databricks notebooks"},
]
response_text = "Databricks notebooks are collaborative, cloud-based environments..."

session_store.save_full_session(
    session_id=test_thread,
    messages=messages,
    response_text=response_text,
    response_time_ms=150.5,
    model_endpoint="databricks-gpt-oss-120b",
)
print(f"✓ save_full_session succeeded for thread {test_thread}")

# Read history back
history = session_store.get_history(test_thread, max_turns=10)
print(f"✓ get_history returned {len(history)} messages")
assert len(history) == 2, f"Expected 2 messages (user + assistant), got {len(history)}"
assert history[0]["role"] == "user"
assert history[1]["role"] == "assistant"
assert "notebooks" in history[0]["content"].lower()
print(f"  [0] {history[0]['role']}: {history[0]['content'][:60]}...")
print(f"  [1] {history[1]['role']}: {history[1]['content'][:60]}...")
print("✓ Round-trip conversation history verified")

# COMMAND ----------

# Test: multi-turn conversation (simulate 3 turns)
for turn in range(2, 4):
    messages = history + [{"role": "user", "content": f"Follow-up question #{turn}"}]
    response = f"Here's the answer to follow-up #{turn}..."
    session_store.save_full_session(
        session_id=test_thread,
        messages=messages,
        response_text=response,
        response_time_ms=100.0 + turn * 10,
    )
    history = session_store.get_history(test_thread, max_turns=10)
    print(f"  Turn {turn}: {len(history)} messages in history")

assert len(history) == 6, f"Expected 6 messages after 3 turns, got {len(history)}"
print(f"✓ Multi-turn conversation: {len(history)} messages after 3 turns")

# COMMAND ----------

# Test: long-term memory via SessionStore
test_user = f"memory-test-{uuid.uuid4().hex[:8]}"

session_store.save_user_memory(test_user, "team", "Platform Engineering")
session_store.save_user_memory(test_user, "focus_area", "Building real-time streaming pipelines with Structured Streaming")
session_store.save_user_memory(test_user, "skill_level", "Advanced Spark user, new to Unity Catalog")

memories = session_store.recall_user_memories(test_user, "what does the user work on", limit=3)
print(f"✓ recall_user_memories returned {len(memories)} results:")
for m in memories:
    print(f"  [{m['key']}]: {m['content']}")

assert len(memories) > 0, "No memories recalled"
print("✓ Long-term memory save + semantic recall verified")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Test cross-session memory persistence
# MAGIC
# MAGIC Simulate what happens when a user returns in a NEW session (different thread_id)
# MAGIC but the same user_id — long-term memories should carry over.

# COMMAND ----------

# New thread, same user
new_thread = f"session-test-{uuid.uuid4().hex[:8]}"

# Old thread has history — new thread should NOT
old_history = session_store.get_history(test_thread, max_turns=10)
new_history = session_store.get_history(new_thread, max_turns=10)
print(f"Old thread ({test_thread}): {len(old_history)} messages")
print(f"New thread ({new_thread}): {len(new_history)} messages")
assert len(new_history) == 0, "New thread should have no history"
print("✓ New thread starts with empty conversation history")

# But long-term memories persist across threads (keyed by user_id, not thread_id)
memories = session_store.recall_user_memories(test_user, "streaming", limit=3)
print(f"\nRecalled memories for returning user:")
for m in memories:
    print(f"  [{m['key']}]: {m['content']}")
assert len(memories) > 0, "Cross-session memories should persist"
print("✓ Long-term memories persist across sessions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✓ All tests passed!
# MAGIC
# MAGIC The Lakebase integration is working:
# MAGIC - **DatabricksStore** connects and creates tables automatically
# MAGIC - **Short-term memory**: conversation history saved/loaded per thread_id
# MAGIC - **Long-term memory**: user facts saved and recalled via semantic search
# MAGIC - **Cross-session**: long-term memories persist when user starts new thread
# MAGIC
# MAGIC Next steps:
# MAGIC - Deploy the updated code via `databricks bundle deploy`
# MAGIC - Re-register the model (Step 4) and redeploy (Step 6)
# MAGIC - Test via the serving endpoint with `thread_id` and `user_id` in `custom_inputs`

# COMMAND ----------

print("=" * 60)
print("LAKEBASE MEMORY INTEGRATION TEST — ALL PASSED")
print("=" * 60)
print(f"  Lakebase project: {LAKEBASE_PROJECT}")
print(f"  Lakebase branch:  {LAKEBASE_BRANCH}")
print(f"  Short-term memory: ✓ (conversation per thread_id)")
print(f"  Long-term memory:  ✓ (user facts with semantic search)")
print(f"  Cross-session:     ✓ (memories persist across threads)")
print("=" * 60)
