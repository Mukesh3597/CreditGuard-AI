CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT DEFAULT (datetime('now','localtime')),

    Age INTEGER,
    Income REAL,
    LoanAmount REAL,
    CreditScore INTEGER,
    MonthsEmployed INTEGER,
    NumCreditLines INTEGER,
    InterestRate REAL,
    LoanTerm INTEGER,
    DTIRatio REAL,

    Education TEXT,
    EmploymentType TEXT,
    MaritalStatus TEXT,
    HasMortgage TEXT,
    HasDependents TEXT,
    LoanPurpose TEXT,
    HasCoSigner TEXT,

    prediction INTEGER,
    default_probability REAL
);
