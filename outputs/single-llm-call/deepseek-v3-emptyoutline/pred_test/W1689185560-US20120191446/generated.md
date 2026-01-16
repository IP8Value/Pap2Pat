Below is the drafted patent application following the provided outline and guidelines. The application is written in formal patent language, avoids references to the research paper, and maintains a standalone structure.

---

# DESCRIPTION  

## THE NAMES OF PARTIES TO A JOINT RESEARCH AGREEMENT  

The present invention was developed as part of an independent research initiative and does not arise from a joint research agreement.  

## FIELD OF THE DISCLOSURE  

The present disclosure relates to the field of computer science, particularly to parsing technologies used in programming languages, data processing, and network communications. More specifically, the invention pertains to a formally verified parsing system based on Parsing Expression Grammars (PEGs), ensuring correctness and termination guarantees through a proof assistant framework.  

## BACKGROUND OF THE DISCLOSURE  

Parsing is a fundamental operation in computer science, essential for tasks such as compilation, data validation, and protocol interpretation. Traditional parsing methods rely on parser generators that produce parsers from context-free grammars (CFGs) or regular expressions. However, these methods suffer from ambiguities, inefficiencies, and a lack of formal correctness guarantees.  

Conventional parser generators, such as Yacc, often produce parsers with inherent bugs or large parsing tables that defy manual verification. While formal verification techniques have been applied to compilers, parsing remains an unverified component in most systems. Parsing Expression Grammars (PEGs) offer an alternative to CFGs by providing unambiguous parsing rules with prioritized choice and greedy repetition. However, existing PEG implementations lack formal guarantees of correctness and termination, particularly when handling left-recursive grammars.  

There exists a need for a parsing system that combines the expressiveness of PEGs with formal verification to ensure total correctness, including termination and adherence to grammar semantics.  

## SUMMARY  

The present invention provides a formally verified parsing system based on Parsing Expression Grammars (PEGs), implemented within a proof assistant framework. The system comprises:  

1. A deep embedding of PEGs in a dependent type system, enabling the formal specification of grammars and semantic actions.  
2. A termination analysis procedure that checks grammar well-formedness, ensuring parsing terminates for all inputs.  
3. A certified interpreter that parses input strings according to the grammar and produces results with correctness guarantees.  

The system supports semantic actions, allowing parsed input to be transformed into structured representations such as abstract syntax trees. By leveraging proof assistant extraction, the system generates executable parsers with formally verified properties, eliminating ambiguities and runtime errors common in conventional parsing tools.  

## DETAILED DESCRIPTION OF ILLUSTRATIVE EMBODIMENTS  

### 1. General Principles of an Embodiment of the Invention  

The invention operates by embedding PEGs in a proof assistant, such as Coq, where grammar rules and semantic actions are expressed as dependent types. The parsing process involves:  

- **Grammar Specification**: A grammar is defined as a set of parsing expressions, including terminals, non-terminals, sequences, choices, repetitions, and semantic actions.  
- **Well-formedness Check**: A static analysis verifies that the grammar is free from left-recursion and other non-terminating constructs.  
- **Certified Interpretation**: A formally verified interpreter executes parsing according to the grammar, ensuring that results conform to the semantics of PEGs.  

The system guarantees termination by enforcing syntactic restrictions on grammars and using a well-founded order for recursive parsing operations.  

### Example 1  

Consider a grammar for mathematical expressions, where non-terminals include `number`, `term`, `factor`, and `expr`. The grammar incorporates semantic actions to evaluate expressions, ensuring that inputs like `(1+2)*(3*4)` yield the correct result, `36`. The parsing process is formally verified to adhere to the grammar's semantics, with termination ensured by the absence of left-recursion.  

### 2. Description of Some Embodiments  

In one embodiment, the system extends PEGs with semantic actions, enabling the construction of parse trees or abstract syntax trees. Semantic actions are embedded in the proof assistant's functional language, ensuring they are terminating and type-safe.  

In another embodiment, the system optimizes parsing performance by:  
- Implementing native support for character ranges (e.g., `[a-z]`) to avoid inefficient derived choices.  
- Using memoization (packrat parsing) to achieve linear time complexity, albeit with increased memory usage.  

### 3. Summary of Three Embodiments of the Invention  

1. **Basic PEG Interpreter**: A certified interpreter for PEGs without semantic actions, focusing on recognition rather than transformation.  
2. **XPEG Interpreter**: An extended interpreter supporting semantic actions, enabling structured output generation.  
3. **Optimized Parser Generator**: A code-generating variant that produces high-performance parsers in target languages (e.g., OCaml or Haskell) while preserving correctness guarantees.  

Each embodiment ensures termination and correctness through formal verification, making the system suitable for critical applications such as compilers, network protocols, and data validation.  

---

This patent application provides a comprehensive and formal description of the invention, adhering to the provided outline and maintaining the required structure and language. Let me know if you'd like any refinements or additional details.