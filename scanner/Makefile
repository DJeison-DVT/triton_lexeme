CC      = gcc
FLEX    = flex
TARGET  = triton_scanner
LEX_SRC = scanner.l
GEN_SRC = lex.yy.c

all: $(TARGET)

$(GEN_SRC): $(LEX_SRC)
	$(FLEX) $(LEX_SRC)

$(TARGET): $(GEN_SRC)
	$(CC) -o $(TARGET) $(GEN_SRC)

clean:
	rm -f $(GEN_SRC) $(TARGET)

test: $(TARGET)
	python3 -m pytest test_scanner.py -v

.PHONY: all clean test
