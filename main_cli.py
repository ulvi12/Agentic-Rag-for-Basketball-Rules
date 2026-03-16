import argparse
import sys
from dotenv import load_dotenv

load_dotenv()

def _print_result(result: dict, trace: bool):
    from src.rag import _format_source_label

    print("\n Answer:\n")
    print(result["answer"])

    if result["sources"]:
        print("\n Sources:")
        for i, meta in enumerate(result["sources"], start=1):
            star = "*" if meta.get("_followed_ref") else ""
            print(f"  [{i}{star}] {_format_source_label(meta)}")

    if trace and "trace" in result:
        t = result["trace"]
        q_type = t.get("question_type", "lookup")
        print("\n Trace:")
        print(f"  Question type  : {q_type}")
        if q_type == "conversational":
            return
        sub_qs = t.get("sub_questions", [])
        router_raw = t.get("router", "—")
        if isinstance(router_raw, list):
            router_display = "override" if all(r == "override" for r in router_raw) else "auto"
        else:
            router_display = "override" if router_raw == "override" else "auto"
        if len(sub_qs) > 1:
            print(f"  Decomposed into: {len(sub_qs)} sub-questions")
            for i, q in enumerate(sub_qs, 1):
                print(f"    {i}. {q}")
        print(f"  Router         : {router_display}")
        print(f"  Leagues queried: {t.get('leagues_queried', [])}")
        if "chunks_after_rerank" in t:
            print(f"  Chunks (rerank): {t['chunks_after_rerank']}")
            print(f"  Refs followed  : {t.get('followed_refs_added', 0)}")
            print(f"  Rerank scores  : {t['rerank_scores']}")
        if "timings" in t:
            tm = t["timings"]
            total = round(sum(tm.values()), 2)
            parts = "  ".join(f"{k}={v}s" for k, v in tm.items())
            print(f" {parts}  [total={total}s]")

def cmd_ingest(args):
    from src.ingestion import ingest
    ingest(force=args.force)

def cmd_chat(args):
    from src.rag import answer
    from src.agents.memory import ConversationMemory

    memory = ConversationMemory()
    print("\n Basketball Rules Chat — type 'quit' to stop, 'clear' to reset memory.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() == "quit":
            print("Goodbye!")
            break
        if question.lower() == "clear":
            memory.clear()
            print("Memory cleared.\n")
            continue

        print("─" * 60)
        result = answer(question, history=memory.get_history(), trace=args.trace)
        _print_result(result, args.trace)
        memory.add(question, result["answer"])

def main():
    parser = argparse.ArgumentParser(
        description="Basketball Rules RAG — ingest rulebooks or chat."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Load all rulebooks into ChromaDB.")
    ingest_parser.add_argument(
        "--force", action="store_true",
        help="Re-ingest even if the collection already has documents."
    )

    chat_parser = subparsers.add_parser("chat", help="Interactive chat with conversation memory.")
    chat_parser.add_argument("--trace", action="store_true", help="Print agent reasoning trace.")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "chat":
        cmd_chat(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()