# DESCRIPTION

## THE NAMES OF PARTIES TO A JOINT RESEARCH AGREEMENT

The names of the parties to the joint research agreement are not applicable in this context, as the invention described herein is the result of individual research and development efforts.

## FIELD OF THE DISCLOSURE

The present disclosure relates to the field of parsing and compiler design, specifically to a formally verified parser interpreter for Parsing Expression Grammars (PEGs). The disclosed invention provides a method and system for generating parsers with total correctness guarantees, ensuring both termination and correctness with respect to the semantics of PEGs.

## BACKGROUND OF THE DISCLOSURE

Parsing is a fundamental process in computer science, essential for tasks ranging from compilation and web development to data security and communication protocols. Traditional approaches to parsing involve the use of parser generators, which take grammars as input and produce parsers. These parsers are typically based on regular expressions (REs) and context-free grammars (CFGs), often expressed in Backus-Naur Form (BNF) syntax. While these tools are widely used, they suffer from several limitations, including potential bugs, large parsing tables that are difficult to inspect, and the lack of formal verification.

Parsing Expression Grammars (PEGs) offer an alternative to CFGs, providing unambiguous and expressive parsing capabilities. PEGs are gaining popularity due to their ease of implementation and ability to handle lexical and syntactical analysis within a single formalism. However, ensuring the correctness and termination of PEG-based parsers remains a significant challenge.

The present invention addresses these challenges by introducing TRX, a PEG-based parser interpreter formally developed in the proof assistant Coq. TRX ensures total correctness guarantees, meaning that the resulting parser is both terminating and correct with respect to its grammar and the semantics of PEGs. This formal verification is achieved through a deep embedding of PEGs in Coq, a reflective procedure for checking grammar well-formedness, and a formally verified interpreter for well-formed PEGs.

## SUMMARY

The present invention provides a method and system for generating formally verified parsers using Parsing Expression Grammars (PEGs). The key features and advantages of the invention include:

1. **Formal Verification**: The parser interpreter is formally developed in the proof assistant Coq, ensuring total correctness guarantees. This means that the generated parser is both terminating and correct with respect to its grammar and the semantics of PEGs.

2. **Deep Embedding**: PEGs are deeply embedded in Coq, allowing for rigorous formal verification of the grammar and the parser interpreter.

3. **Well-Formedness Check**: A certified algorithm is developed to verify that a given PEG is well-formed, ensuring that the grammar is complete and does not contain left-recursive rules that could lead to non-termination.

4. **Interpreter Development**: A formally verified interpreter for well-formed PEGs is developed, ensuring that the parser correctly interprets the grammar and produces the expected parsing results.

5. **Code Extraction**: The formal development in Coq allows for the extraction of a certified parser in target languages such as OCaml, Haskell, or Scheme, making the technology accessible to a broader audience.

6. **Practical Performance**: The invention includes optimizations to improve the performance of the extracted parser, making it suitable for real-world applications.

7. **Scalability**: The system is designed to handle large and complex grammars, as demonstrated by case studies involving the parsing of XML and Java.

8. **Error Handling**: Future work aims to enhance the system with support for error messages, improving the usability of the parser.

The invention represents a significant advancement in the field of parsing and compiler design, providing a robust and reliable solution for generating parsers with formal correctness guarantees.

## DETAILED DESCRIPTION OF ILLUSTRATIVE EMBODIMENTS

### 1. Generals Principles of An Embodiment of the Invention

The present invention, TRX, is a formally verified parser interpreter for Parsing Expression Grammars (PEGs). The core principles of TRX include:

- **Formal Verification**: The entire development of TRX is conducted within the Coq proof assistant, ensuring that all definitions, theorems, and proofs are formally verified. This guarantees that the generated parser is both terminating and correct with respect to its grammar and the semantics of PEGs.

- **Deep Embedding**: PEGs are deeply embedded in Coq, allowing for a rigorous formalization of the grammar and its properties. This deep embedding ensures that the grammar is well-defined and can be checked for well-formedness.

- **Well-Formedness Check**: A certified algorithm is developed to verify that a given PEG is well-formed. This check ensures that the grammar is complete and does not contain left-recursive rules that could lead to non-termination. The well-formedness check is performed using a reflective procedure that iterates over the grammar and applies a set of inference rules to determine the properties of the parsing expressions.

- **Interpreter Development**: A formally verified interpreter for well-formed PEGs is developed. The interpreter is defined as a function in Coq that takes a well-formed PEG and an input string and produces the parsing result. The interpreter is proven to be correct with respect to the semantics of PEGs, ensuring that it correctly interprets the grammar and produces the expected parsing results.

- **Code Extraction**: The formal development in Coq allows for the extraction of a certified parser in target languages such as OCaml, Haskell, or Scheme. This extraction process discards the logical reasoning and proofs, leaving only the computational content, resulting in a certified parser that can be used in practical applications.

- **Optimizations**: The invention includes several optimizations to improve the performance of the extracted parser. These optimizations include fixing issues with the `rev` function, implementing the range operator natively, and optimizing the comparison of characters. Additionally, the performance of the parser can be further improved by tweaking the parameters of the target language's garbage collector.

- **Scalability**: TRX is designed to handle large and complex grammars, as demonstrated by case studies involving the parsing of XML and Java. The well-formedness check and the interpreter are scalable and can be applied to grammars of varying sizes and complexities.

- **Future Enhancements**: Future work aims to extend TRX with support for error messages, improve the performance of the parser, and develop a parser generator that can synthesize PEGs from textual descriptions, making the technology more accessible to users who are not familiar with Coq.

### Example 1

Consider a simple PEG for parsing mathematical expressions. The grammar includes rules for white space, numbers, terms, factors, and expressions. The grammar is defined as follows:

- `ws` consumes all white space from the beginning of the input.
- `number` reads a sequence of digits.
- `term` and `factor` handle multiplication and division.
- `expr` handles addition and subtraction.

The grammar is expressed in Coq as follows:

```coq
Parameter prod : Type := Inductive prod := ws | number | term | factor | expr.

Parameter prod_type : prod -> Type := fun p => match p with
  | ws => unit
  | number => nat
  | term => nat
  | factor => nat
  | expr => nat
end.

Parameter production : forall p : prod, PExp (prod_type p) := fun p => match p with
  | ws => [•]* [ws]
  | number => [0-9]+ [ws] [→] digListToNat
  | term => factor ([*] factor)* [ws] [→] mul
  | factor => number / ('(' [ws] expr [ws] ')') [ws]
  | expr => term ([+] term)* [ws] [→] add
end.

Parameter start : prod := expr.
```

In this example, the `digListToNat` function converts a list of digits to a natural number, and the `mul` and `add` functions handle multiplication and addition, respectively. The grammar is well-formed and can be checked using the well-formedness algorithm. The interpreter can then be used to parse input strings and produce the expected parsing results.

### 2. Description of Some Embodiments

#### 2.1. Deep Embedding of PEGs in Coq

PEGs are deeply embedded in Coq, allowing for a rigorous formalization of the grammar and its properties. The deep embedding ensures that the grammar is well-defined and can be checked for well-formedness. The embedding is achieved by defining the set of parsing expressions and the grammar structure in Coq.

The set of parsing expressions, `PExp`, is defined as an inductive family indexed by the type of the semantic value associated with the expression. The typing rules for `PExp` are given in Figure 3 of the research paper. The grammar structure is defined as a tuple `(V_T, V_N, P_type, P_exp, v_start)`, where `V_T` is the set of terminals, `V_N` is the set of non-terminals, `P_type` is the interpretation of the types of the semantic values, `P_exp` is the interpretation of the productions, and `v_start` is the starting production.

#### 2.2. Well-Formedness Check

The well-formedness check is a critical component of TRX, ensuring that the grammar is complete and does not contain left-recursive rules that could lead to non-termination. The check is performed using a reflective procedure that iterates over the grammar and applies a set of inference rules to determine the properties of the parsing expressions.

The properties of parsing expressions are classified into three groups: "0" (can succeed without consuming any input), "> 0" (can succeed after consuming some input), and "⊥" (can fail). The inference rules for deriving these properties are given in Figure 5 of the research paper. The well-formedness check is performed by iterating over the expression set of the grammar and applying the inference rules until reaching a fix-point.

#### 2.3. Interpreter Development

The interpreter for well-formed PEGs is developed in Coq and is proven to be correct with respect to the semantics of PEGs. The interpreter is defined as a function that takes a well-formed PEG and an input string and produces the parsing result. The function performs pattern matching on the parsing expression and interprets it according to the semantics of PEGs.

The termination argument for the interpreter is based on the decrease of the pair of arguments `(e, s)` in recursive calls with respect to a well-founded relation. The relation is defined such that `(e_1, s_1)` is bigger than `(e_2, s_2)` if the step-count in the semantics of the first pair is greater than that of the second pair. The well-foundedness of the relation ensures that all recursive calls are indeed decreasing, guaranteeing the termination of the interpreter.

#### 2.4. Code Extraction

The formal development in Coq allows for the extraction of a certified parser in target languages such as OCaml, Haskell, or Scheme. The extraction process discards the logical reasoning and proofs, leaving only the computational content. The extracted parser can be used in practical applications, providing a certified and reliable solution for parsing.

### 3. Summary of Three Embodiments of the Invention

#### Embodiment 1: Formal Verification and Deep Embedding

In the first embodiment, the invention provides a method for generating formally verified parsers using PEGs. The method involves deeply embedding PEGs in the Coq proof assistant and developing a certified algorithm to check the well-formedness of the grammar. The well-formedness check ensures that the grammar is complete and does not contain left-recursive rules that could lead to non-termination. The interpreter for well-formed PEGs is developed in Coq and is proven to be correct with respect to the semantics of PEGs. The formal development in Coq allows for the extraction of a certified parser in target languages such as OCaml, Haskell, or Scheme.

#### Embodiment 2: Optimized Performance

In the second embodiment, the invention includes several optimizations to improve the performance of the extracted parser. These optimizations include fixing issues with the `rev` function, implementing the range operator natively, and optimizing the comparison of characters. Additionally, the performance of the parser can be further improved by tweaking the parameters of the target language's garbage collector. The optimized parser is suitable for real-world applications and demonstrates reasonable performance, as demonstrated by case studies involving the parsing of XML and Java.

#### Embodiment 3: Scalability and Future Enhancements

In the third embodiment, the invention is designed to handle large and complex grammars, as demonstrated by case studies involving the parsing of XML and Java. The well-formedness check and the interpreter are scalable and can be applied to grammars of varying sizes and complexities. Future work aims to extend TRX with support for error messages, improve the performance of the parser, and develop a parser generator that can synthesize PEGs from textual descriptions, making the technology more accessible to users who are not familiar with Coq.

The invention represents a significant advancement in the field of parsing and compiler design, providing a robust and reliable solution for generating parsers with formal correctness guarantees.