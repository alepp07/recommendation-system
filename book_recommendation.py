import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QSlider, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PyQt6.QtCore import Qt
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from itertools import islice


class BookAnalysisApp(QWidget):
    def __init__(self):
        super().__init__()

        self.books = pd.read_csv("Books.csv")
        self.ratings = pd.read_csv("Ratings.csv")
        self.users = pd.read_csv("Users.csv")

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Book Analysis App')
        self.setGeometry(100, 100, 800, 600)

        # Set the window to be maximized on startup
        # self.showMaximized()

        self.btn_info = QPushButton('Show DataFrame Info', self)
        self.btn_info.clicked.connect(self.show_data_info)

        self.create_widgets()
        self.create_layout()

        # Automatically show the "Number of Publications for Every Year" graph on widget opening
        self.btn_popularity.setChecked(True)
        self.plot_publications()

    def create_widgets(self):
        # Create buttons
        self.btn_popularity = QPushButton('Number of Publications', self)
        self.btn_popularity.clicked.connect(self.plot_publications)

        self.btn_publishers = QPushButton('Top 10 Publishers', self)
        self.btn_publishers.clicked.connect(self.plot_top_publishers)

        # Create a new button for the high-rated books graph
        self.btn_high_rating = QPushButton('High-Rated Books', self)
        self.btn_high_rating.clicked.connect(self.plot_high_rated_books)

        # Create QLineEdit for search term
        self.search_edit = QLineEdit(self)
        self.search_edit.setPlaceholderText("Enter Author or Book Name")

        # Create a new button for the high-rated books graph
        self.btn_give_books = QPushButton('Give Books', self)
        self.btn_give_books.clicked.connect(self.plot_give_books)

        # Create QSlider for year filtering
        self.year_slider = QSlider(Qt.Orientation.Horizontal)
        self.year_slider.setMinimum(1950)
        self.year_slider.setMaximum(2023)
        self.year_slider.setValue(2023)  # Set default value to 2023
        self.year_slider.valueChanged.connect(self.update_year_label)

        # Create QSlider for rating filtering
        self.rating_slider = QSlider(Qt.Orientation.Horizontal)
        self.rating_slider.setMinimum(1)
        self.rating_slider.setMaximum(10)
        self.rating_slider.setValue(10)  # Set default value to 10
        self.rating_slider.valueChanged.connect(self.update_rating_label)

        # Create QLabel for slider value display
        self.year_label = QLabel('Selected Year: 2023', self)
        self.rating_label = QLabel('Selected Rating: 10', self)

        # Create QPushButton for search and recommendation
        self.btn_search = QPushButton('Search and Recommend', self)
        self.btn_search.clicked.connect(self.search_and_recommend)

        # Create matplotlib figure and canvas
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)

    def create_layout(self):
        # Create layouts
        main_layout = QVBoxLayout(self)

        # Create top layout for the slider and search button on the right
        top_layout = QHBoxLayout()

        # Create bottom left layout for all buttons except "Search and Recommend"
        bottom_left_layout = QVBoxLayout()
        bottom_left_layout.addWidget(self.btn_popularity)
        bottom_left_layout.addWidget(self.btn_publishers)
        bottom_left_layout.addWidget(self.btn_high_rating)

        # Move the button for dataframe information to the center-bottom
        center_bottom_layout = QHBoxLayout()
        center_bottom_layout.addWidget(self.btn_info)

        # Add the button to the center-bottom layout
        main_layout.addLayout(center_bottom_layout)

        # Add the buttons to the top-left corner
        top_layout.addLayout(bottom_left_layout)

        # Move the search input field beside the "Top Publications" button
        bottom_left_layout.addWidget(self.search_edit)
        bottom_left_layout.addWidget(self.rating_label)
        bottom_left_layout.addWidget(self.rating_slider)
        bottom_left_layout.addWidget(self.btn_give_books)


        # Create a horizontal layout for the slider and search button
        slider_search_layout = QVBoxLayout()
        slider_search_layout.addWidget(self.year_label)
        slider_search_layout.addWidget(self.year_slider)
        slider_search_layout.addWidget(self.btn_search)

        # Add the slider and search button to the top-right corner
        top_layout.addStretch(1)  # Add stretch to push to the right
        top_layout.addLayout(slider_search_layout)

        # Add the top layout to the main layout
        main_layout.addLayout(top_layout)

        # Create bottom right layout for the graph
        bottom_right_layout = QVBoxLayout()
        bottom_right_layout.addWidget(self.canvas)

        # Add the combined layout to the main layout
        main_layout.addLayout(bottom_right_layout)

        # Set layout to the main layout
        self.setLayout(main_layout)

    def show_data_info(self):
        info_text = "Books DataFrame Info:\n" + str(self.books.info()) + "\n\n"
        info_text += "Ratings DataFrame Info:\n" + str(self.ratings.info()) + "\n\n"
        info_text += "Users DataFrame Info:\n" + str(self.users.info())

        info_label = QLabel(info_text)
        info_label.setWindowTitle('DataFrame Information')
        info_label.setGeometry(100, 100, 600, 400)
        info_label.show()
    def plot_high_rated_books(self):
        high_rated_books = self.ratings[self.ratings['Book-Rating'] >= 5]

        # Merge with book information
        high_rated_books_with_info = pd.merge(high_rated_books, self.books, on="ISBN", how="inner")

        self.ax.clear()

        # Calculate the popularity and average rating for each book
        book_stats = high_rated_books_with_info.groupby('Book-Title').agg(
            {'User-ID': 'count', 'Book-Rating': 'mean'}).reset_index()

        # Sort books by popularity and then by average rating
        book_stats = book_stats.sort_values(by=['User-ID', 'Book-Rating'], ascending=False)

        # Plot the top 10 high-rated books based on popularity and rating
        sns.barplot(x='User-ID', y='Book-Title', data=book_stats.head(10), ax=self.ax, palette='viridis')
        self.ax.set_title("Top 10 High-Rated Books (Popularity and Rating)")
        self.ax.set_xlabel("Number of High Ratings")
        self.ax.set_ylabel("Book Title")

        self.canvas.draw()

    def plot_publications(self):
        valid_years = self.books['Year-Of-Publication'].astype(str).str.isnumeric()
        books = self.books[valid_years]

        books['Year-Of-Publication'] = books['Year-Of-Publication'].astype(int)

        start_year = 1950
        current_year = pd.Timestamp.now().year
        books = books[(books['Year-Of-Publication'] >= start_year) & (books['Year-Of-Publication'] <= current_year)]

        self.ax.clear()
        self.ax.set_title("Number of publications for each year")

        # Use seaborn barplot to automatically handle x-axis labels
        num_years = current_year - start_year + 1
        colors = sns.color_palette("Reds", n_colors=num_years)

        sns.countplot(x='Year-Of-Publication', data=books, ax=self.ax, palette=colors)

        # Update x-axis ticks to show every 2 years
        years_to_display = range(start_year, current_year + 1, 2)
        self.ax.set_xticks([year - start_year for year in years_to_display])
        self.ax.set_xticklabels(years_to_display, rotation=45, ha='right')

        self.canvas.draw()

    def plot_top_publishers(self):
        ratings_with_name = pd.merge(self.ratings, self.books, on="ISBN", how="inner")
        top_publishers = ratings_with_name["Publisher"].value_counts().head(10)

        self.ax.clear()
        top_publishers.plot(kind="bar", grid=True, ax=self.ax)
        self.ax.set_title("Top 10 publishers with most books")
        self.ax.set_xlabel("Publisher")
        self.ax.set_ylabel("Number of books")

        # Rotate x-axis labels for better readability and reduce font size
        self.ax.set_xticklabels(self.ax.get_xticklabels(), rotation=45, ha='right', fontsize=6)

        self.canvas.draw()

    def update_year_label(self, value):
        self.year_label.setText(f'Selected Year: {value}')

    def update_rating_label(self, value):
        self.rating_label.setText(f'Selected Rating: {value}')

    def plot_give_books(self):
        max_rating = self.rating_slider.value()

        # Merge the dataframes
        df1 = self.users.merge(self.ratings, on='User-ID')
        data = df1.merge(self.books, on='ISBN')

        # Preprocess the data
        df = data.copy()
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.drop(columns=["ISBN", "Year-Of-Publication", "Image-URL-S", "Image-URL-M"], axis=1, inplace=True)
        df.drop(index=df[df["Book-Rating"] == 0].index, inplace=True)
        df["Book-Title"] = df["Book-Title"].apply(lambda x: re.sub("[\W_]+", " ", x).strip())

        # Filter users who voted more than 200 times
        new_df = df[df['User-ID'].map(df['User-ID'].value_counts()) > 200]

        # Create TF-IDF matrix for book titles
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(new_df['Book-Title'])

        # Initialize Nearest Neighbors model
        knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6)

        # Fit the model with the TF-IDF matrix
        knn_model.fit(tfidf_matrix)

        search_book = self.search_edit.text()

        recommend_books = set(f"{title} by {author} (rating: {rating})" for title, author, rating in
                              zip(new_df['Book-Title'], new_df['Book-Author'], new_df['Book-Rating']) if
                              (search_book.lower() in title.lower() or search_book.lower() in author.lower()) and
                              rating <= max_rating)

        recommend_books = sorted(recommend_books, key=lambda x: float(x.split('(rating: ')[-1].rstrip(')')),
                                 reverse=True)

        # Limit the set to 30 items
        recommend_books = list(islice(recommend_books, 30))

        # Extract titles, authors, and ratings
        titles = [book.split(' by ')[0] for book in recommend_books]
        authors = [book.split(' by ')[1].split(' (rating: ')[0] for book in recommend_books]
        ratings = [float(book.split('(rating: ')[-1].rstrip(')')) for book in recommend_books]

        # Plotting
        self.ax.clear()
        self.ax.figure.set_size_inches(10, 6)  # Set the figure size using self.ax.figure
        self.ax.barh(titles, ratings, color='skyblue')  # Use barh for horizontal bar plot
        self.ax.set_ylabel('Book Title')  # Set y-axis label
        self.ax.set_xlabel('Rating')  # Set x-axis label
        self.ax.set_title(f'{search_book} Books Ratings')

        # Increase space for y-axis label
        self.ax.figure.subplots_adjust(left=0.7)  # Adjust the left margin to allocate more space for the y-axis label

        # Display the plot on the canvas
        self.canvas.draw()

    def search_and_recommend(self):
        search_term = self.search_edit.text()
        selected_year = self.year_slider.value()

        # Filter books based on the selected year
        books_in_year = self.books[self.books['Year-Of-Publication'] == selected_year]

        if books_in_year.empty:
            self.ax.clear()
            self.ax.set_title(f"No books released in {selected_year}")
            self.canvas.draw()
            return

        # Merge with book information
        books_with_ratings = pd.merge(self.ratings, books_in_year, on="ISBN", how="inner")

        # Group by book title and calculate the average rating
        avg_ratings = books_with_ratings.groupby('Book-Title')['Book-Rating'].mean().reset_index()

        # Sort by average rating in descending order
        avg_ratings = avg_ratings.sort_values(by='Book-Rating', ascending=False)

        # Display top 5 recommendations
        self.ax.clear()
        sns.barplot(x='Book-Rating', y='Book-Title', data=avg_ratings.head(5), ax=self.ax, palette='Blues_d')
        self.ax.set_title(f"Top 5 Recommended Books in {selected_year}")
        self.ax.set_xlabel("Average Rating")
        self.ax.set_ylabel("Book Title")

        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = BookAnalysisApp()
    ex.show()
    sys.exit(app.exec())
