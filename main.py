#!/usr/bin/env python
from __future__ import annotations

from collections import deque, namedtuple
from itertools import product

from terminaltables import AsciiTable


class TokenType:
    IDENTIFIER = 0
    AND = 1
    OR = 2
    NOT = 3
    LEFT_PAREN = 4
    RIGHT_PAREN = 5
    EOF = 6

    __names = {
        IDENTIFIER: "IDENTIFIER",
        AND: "AND",
        OR: "OR",
        NOT: "NOT",
        LEFT_PAREN: "LEFT_PAREN",
        RIGHT_PAREN: "RIGHT_PAREN",
        EOF: "EOF",
    }

    OPERATORS = (AND, OR, NOT)

    @classmethod
    def get_name(cls, tk_type):
        return cls.__names.get(tk_type, '')

op_precedence = {
    TokenType.NOT: 3,
    TokenType.AND: 2,
    TokenType.OR: 1,
}

op_left_assoc = {
    TokenType.NOT: False,
    TokenType.AND: True,
    TokenType.OR: True,
}


keywords = {
    'and': TokenType.AND, # &
    'or': TokenType.OR, # |
    'not': TokenType.NOT # !
}


class Token(namedtuple('Token', ('tk_type', 'lexeme'))):
    def __repr__(self):
        return f"Token( type={TokenType.get_name(self.tk_type)}, lexeme='{self.lexeme}' )"


class Scanner:
    def __init__(self, source):
        self.source = source
        self.len_source = len(source)
        self.tokens = [] # type: list[Token]

        self.start = 0
        self.next_unobserved = 0

    def is_at_end(self):
        return self.next_unobserved >= self.len_source

    def peek(self):
        return self.source[self.next_unobserved]        

    def advance(self):
        ch = self.source[self.next_unobserved]
        self.next_unobserved += 1
        return ch

    def advance_if_match(self, expected):
        if self.is_at_end(): return False
        ch = self.peek()
        if ch == expected:
            self.advance()
            return True

        return False

    def add_token(self, tk_type):
        lexeme = self.source[self.start:self.next_unobserved]
        self.tokens.append(Token(tk_type, lexeme))

    def scan_all(self):
        while not self.is_at_end():
            ch = self.advance()
            if ch == '!':
                self.add_token(TokenType.NOT)
            elif ch == '&':
                self.add_token(TokenType.AND)
            elif ch == '(':
                self.add_token(TokenType.LEFT_PAREN)
            elif ch == ')':
                self.add_token(TokenType.RIGHT_PAREN)
            elif ch == '|':
                self.add_token(TokenType.OR)

            elif ch == ' ' or ch == '\t' or ch == '\n':
                pass 
            else:
                if ch.isalpha():
                    # while not self.is_at_end() and not self.peek().isalpha():
                    #   self.advance()
                    while True:
                        if (self.is_at_end()):
                            break

                        l_next = self.peek()
                        if l_next.isalpha():
                            self.advance()
                        else:
                            break

                    text = self.source[self.start:self.next_unobserved]
                    tk_type = keywords.get(text, TokenType.IDENTIFIER)
                    self.add_token(tk_type)

                else:
                    # error
                    pass

            self.start = self.next_unobserved

    def get_tokens(self):
        return self.tokens


def oper_and(p, q):
    return 1 if p == 1 and q == 1 else 0

def oper_or(p, q):
    return 0 if p == 0 and q == 0 else 1

def oper_not(p):
    return 0 if p == 1 else 1

def found_lower(stack_top_type, current_type):
    stack_top_prec = op_precedence[stack_top_type]
    current_prec = op_precedence[current_type]
    
    if stack_top_prec > current_prec:
        return True

    if stack_top_prec == current_prec and op_left_assoc[stack_top_type]:
        return True

    return False

class Parser:

    def __init__(self):
        self.postfix_tokens = [] # type: list[Token]
        self.identifier_lexemes = []

    def get_variables(self):
        return self.identifier_lexemes

    def iter_rows(self):
        for v in product((0, 1), repeat=len(self.identifier_lexemes)):
            yield v + (self.execute(v),)

    def parse(self, tokens: list[Token]):
        self.postfix_tokens = []
        self.identifier_lexemes = []

        output_q = [] # type: list[Token]
        op_stack = deque() # type: list[Token]

        for token in tokens:
            tk_type = token.tk_type
            if tk_type == TokenType.IDENTIFIER:

                if not token.lexeme in self.identifier_lexemes:
                    self.identifier_lexemes.append(token.lexeme)

                output_q.append(token)

            elif tk_type == TokenType.LEFT_PAREN:
                op_stack.append(token)


            elif tk_type == TokenType.RIGHT_PAREN:
                while True:
                    if len(op_stack) == 0:
                        break
                    if op_stack[-1].tk_type == TokenType.LEFT_PAREN:
                        op_stack.pop();
                        break

                    output_q.append(op_stack.pop())


            elif tk_type in TokenType.OPERATORS:
                while True:
                    if len(op_stack) == 0:
                        break

                    top = op_stack[-1]
                    if top.tk_type == TokenType.LEFT_PAREN:
                        break

                    if found_lower(top.tk_type, tk_type):
                        output_q.append(op_stack.pop())
                    else:
                        break

                op_stack.append(token)

        while len(op_stack) > 0:
            output_q.append(op_stack.pop())

        self.postfix_tokens = output_q


    def execute(self, vals: tuple[int]):
        operand_stack = deque()
        for token in self.postfix_tokens:
            if token.tk_type == TokenType.IDENTIFIER:
                val = vals[self.identifier_lexemes.index(token.lexeme)]
                operand_stack.append(val)

            elif token.tk_type == TokenType.AND:
                q = operand_stack.pop()
                p = operand_stack.pop()
                res = oper_and(p, q)
                operand_stack.append(res)

            elif token.tk_type == TokenType.OR:
                q = operand_stack.pop()
                p = operand_stack.pop()
                res = oper_or(p, q)
                operand_stack.append(res)

            elif token.tk_type == TokenType.NOT:
                p = operand_stack.pop()
                res = oper_not(p)
                operand_stack.append(res)


        return operand_stack.pop()


class TruthTableGenerator:
    def __init__(self, source):
        self.source = source
        self.scanner = Scanner(source)
        self.parser = Parser()

        self.variables = []

    def process(self):
        self.scanner.scan_all()
        self.parser.parse(self.scanner.get_tokens())
        self.variables = self.parser.get_variables()

    def display_table(self):

        headers = self.variables + [self.source]
        table_data = [headers]

        for row in self.parser.iter_rows():
            table_data.append(row)

        table = AsciiTable(table_data)
        print(table.table)


gen = TruthTableGenerator("(p|q)&!(p&q)")
gen.process()
gen.display_table()

# sc = Scanner('(p or q) and not (p and q)')
# sc.scan_all()

# rn = Runner(sc.get_tokens())
# rn.run()

