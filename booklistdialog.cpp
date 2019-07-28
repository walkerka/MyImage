#include "booklistdialog.h"
#include <QListWidget>
#include <QMenu>
#include <QLineEdit>
#include <QAction>
#include <QFormLayout>
#include <QPushButton>
#include <QScrollArea>
#include <QLabel>
#include <QMessageBox>
#include "book.h"

BookListDialog::BookListDialog(QWidget *parent) :
    QWidget(parent),
    mId(0)
{
    mBooksField = new QListWidget();
    mNameLineEdit = new QLineEdit();
    mKeyLineEdit = new QLineEdit();
    mActionDelete = new QAction(this);

    QPushButton* mOpenButton;
    QPushButton* mAddButton;
    QPushButton* mDeleteButton;
    QPushButton* mCancelButton;
    mOpenButton = new QPushButton(tr("Open"));
    mAddButton = new QPushButton(tr("Add"));
    mDeleteButton = new QPushButton(tr("Delete"));
    mCancelButton = new QPushButton(tr("Cancel"));


    QFormLayout* form = new QFormLayout();
    form->addRow(tr("Name"), mNameLineEdit);
    form->addRow(tr("Key"), mKeyLineEdit);

    QVBoxLayout* l = new QVBoxLayout();
    l->addLayout(form);
    l->addWidget(mBooksField, 1);

    QHBoxLayout* hbox = new QHBoxLayout();
    hbox->addWidget(mOpenButton);
    hbox->addWidget(mAddButton);
    hbox->addWidget(mDeleteButton);
    hbox->addWidget(mCancelButton);
    l->addLayout(hbox);
    QLabel* previewField = new QLabel();
    l->addWidget(previewField);
    mPreviewField = previewField;
    setLayout(l);

    connect(mNameLineEdit, SIGNAL(textChanged(QString)), this, SLOT(updateBooks()));
    connect(mKeyLineEdit, SIGNAL(textChanged(QString)), this, SLOT(updateBooks()));
    connect(mBooksField, SIGNAL(currentItemChanged(QListWidgetItem*,QListWidgetItem*)),
            this, SLOT(onSelect(QListWidgetItem*,QListWidgetItem*)));
    connect(mBooksField, SIGNAL(itemDoubleClicked(QListWidgetItem*)),
            this, SLOT(onBookDoubleClick(QListWidgetItem*)));
    connect(mBooksField, SIGNAL(customContextMenuRequested(QPoint)),
            this, SLOT(onContextMenu(QPoint)));
    connect(mActionDelete, SIGNAL(toggled(bool)), this, SLOT(deleteBooks()));
    connect(mOpenButton, SIGNAL(clicked(bool)), this, SLOT(onBookOpen()));
    connect(mAddButton, SIGNAL(clicked(bool)), this, SLOT(onBookAdd()));
    connect(mDeleteButton, SIGNAL(clicked(bool)), this, SLOT(deleteBooks()));
    connect(mCancelButton, SIGNAL(clicked(bool)), this, SLOT(hide()));
}

BookListDialog::~BookListDialog()
{
}

void BookListDialog::open()
{
    updateBooks();
    show();
}

void BookListDialog::updateBooks()
{
    if (!Archive::GetInstance())
    {
        return;
    }
    QList<int> bookIds;
    QStringList names;
    Book::SearchBooks(mNameLineEdit->text(), mKeyLineEdit->text(), bookIds, names);
    mBooksField->clear();
    for (int i = 0; i < bookIds.size(); ++i)
    {
        int id = bookIds[i];
        const QString& name = names[i];
        QListWidgetItem* item = new QListWidgetItem(name);
        item->setData(Qt::UserRole, QVariant(id));
        mBooksField->addItem(item);
    }
}


void BookListDialog::onBookOpen()
{
    QList<QListWidgetItem*> items = mBooksField->selectedItems();
    if (items.size() > 0)
    {
        int id = items.first()->data(Qt::UserRole).toInt();
        emit openBook(id);
        hide();
    }
}

void BookListDialog::onBookAdd()
{
    emit addBook();
}

void BookListDialog::deleteBooks()
{
    if (QMessageBox::warning(this, tr("Delete"), tr("Delete selected items?"), QMessageBox::Yes|QMessageBox::No) == QMessageBox::Yes)
    {
        QList<QListWidgetItem*> items = mBooksField->selectedItems();
        foreach (QListWidgetItem* item, items)
        {
            int id = item->data(Qt::UserRole).toInt();
            Book::DeleteBook(id);
        }
        updateBooks();
    }
}

void BookListDialog::onSelect(QListWidgetItem* current, QListWidgetItem*)
{
    int id = 0;
    if (current)
    {
        id = current->data(Qt::UserRole).toInt();
    }
    mId = id;

    int imgId = 0;
    if (id)
    {
        imgId = Book::GetBookHomePage(id);
    }

    if (imgId)
    {
        Image img(imgId);
        QImage::Format format = QImage::Format_RGBA8888;
        switch (img.GetPixelSize())
        {
        case 1:
            format = QImage::Format_Grayscale8;
            break;
        case 3:
            format = QImage::Format_RGB888;
            break;
        case 4:
        default:
            break;
        }
        QImage mResult(img.GetData(),img.GetWidth(),img.GetHeight(), img.GetWidth() * img.GetPixelSize(),format);
        int w = img.GetWidth();
        int h = img.GetHeight();
        int maxS = 300;
        if (w > h)
        {
            if (w > maxS)
            {
                h = h * maxS / w;
                w = maxS;
            }
        }
        else
        {
            if (h > maxS)
            {
                w = w * maxS / h;
                h = maxS;
            }
        }
        mPreviewField->setPixmap(QPixmap::fromImage(mResult.mirrored().scaled(w, h)));
        mPreviewField->resize(w, h);
    }
    else
    {
        mPreviewField->setPixmap(QPixmap());
        mPreviewField->resize(1, 1);
    }
}

void BookListDialog::onBookDoubleClick(QListWidgetItem* current)
{
    if (current)
    {
        int id = current->data(Qt::UserRole).toInt();
        mId = id;
        emit openBook(id);
        hide();
    }
}

void BookListDialog::onContextMenu(const QPoint& pos)
{
    QListWidgetItem* item = mBooksField->itemAt(pos);
    if (!item)
    {
        return;
    }

    QMenu *popMenu = new QMenu(this);
    popMenu->addAction(mActionDelete);
    popMenu->exec(QCursor::pos());
}
