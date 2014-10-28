#-------------------------------------------------
#
# Project created by QtCreator 2014-10-28T10:25:35
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = ekstrasi
TEMPLATE = app


SOURCES += main.cpp

INCLUDEPATH += C:\opencv-mingw\install\include
LIBS += -L"C:/opencv-mingw/install/x64/mingw/bin"
LIBS += -lopencv_core249 -lopencv_highgui249 -lopencv_imgproc249
