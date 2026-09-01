from model import train

if __name__ == "__main__":
    result = train()
    winner = result["winner"]
    print(f"Saved {winner['rule_count']} rules; objective={winner['objective']:.3f}; runtime={result['runtime_seconds']:.2f}s")
