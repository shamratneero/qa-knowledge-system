from data_loader import load_knowledge_base


def main():
    df = load_knowledge_base()

    print(df.head())

    print()

    print(f"Loaded {len(df)} records.")


if __name__ == "__main__":
    main()
