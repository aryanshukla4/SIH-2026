* Canonical small MPS example. Hand-verifiable: 3 rows, 3 cols, 6 nonzeros.
NAME          TESTLP
ROWS
 N  COST
 L  LIM1
 G  LIM2
 E  MYEQN
COLUMNS
    XONE      COST         1.0   LIM1         1.0
    XONE      LIM2         1.0
    YTWO      COST         2.0   LIM1         1.0
    YTWO      MYEQN       -1.0
    ZTHREE    COST         3.0   LIM2         1.0
    ZTHREE    MYEQN        1.0
RHS
    RHS       LIM1         4.0   LIM2         1.0
    RHS       MYEQN        7.0
BOUNDS
 UP BND       XONE         4.0
 LO BND       YTWO        -1.0
 UP BND       YTWO         1.0
ENDATA
