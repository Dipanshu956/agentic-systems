from build_chain import build_chain


def is_response_valid(response: str) -> tuple[bool, list[str]]:
    errors = []

    if not isinstance(response, str):
        errors.append("Response must be a string.")
        return False, errors

    if not response.strip():
        errors.append("Response must not be empty after trimming spaces.")

    word_count = len(response.split())

    if word_count > 100:
        errors.append(
            f"Response must not exceed 100 words. Current count: {word_count}"
        )

    return len(errors) == 0, errors


def main():
    chain = build_chain()

    test_cases = [
        {
            "topic": "LangChain Expression Language",
            "analogy_domain": "school assembly line",
        },
        {
            "topic": "Prompt Templates",
            "analogy_domain": "wedding invitation cards",
        },
        {
            "topic": "Output Parsers",
            "analogy_domain": "food delivery packaging",
        },
    ]

    for case in test_cases:
        print("\n" + "=" * 60)

        response = chain.invoke(case)

        valid, errors = is_response_valid(response)

        print("Input Dictionary:")
        print(case)

        print("\nGenerated Response:")
        print(response)

        print("\nValidation Result:")
        print(valid)

        if errors:
            print("\nValidation Errors:")
            for error in errors:
                print("-", error)

        print("=" * 60)


if __name__ == "__main__":
    main()