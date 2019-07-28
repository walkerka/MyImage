CREATE TABLE book (
id integer primary key autoincrement,
name text,
keywords text,
createDate datetime,
homePageId integer
);

CREATE TABLE image (
id integer primary key autoincrement,
width integer,
height integer,
name text,
imageData blob
);

CREATE TABLE bookPage (
bookId integer not null,
pageId integer not null,
imageId integer not null
);
