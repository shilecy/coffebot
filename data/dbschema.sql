-- ZUS Coffee Outlet Schema
CREATE TABLE outlets (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    address TEXT NOT NULL,
    hours TEXT,
    services TEXT
);