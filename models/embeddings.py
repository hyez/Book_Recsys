import torch
import torch.nn as nn
import torch.nn.functional as F


class item(torch.nn.Module):
    def __init__(self, config):
        super(item, self).__init__()
        self.num_books = config.num_books
        self.embedding_dim = config.embedding_dim


        # self.embedding_books = torch.nn.Linear(
        #     in_features=self.num_books,
        #     out_features=self.embedding_dim,
        #     bias=False
        # )

        self.embedding_books = torch.nn.Embedding(
            num_embeddings=self.num_books,
            embedding_dim=self.embedding_dim
        )
        
    def forward(self, book_idx, vars=None):
        book_emb = self.embedding_books(book_idx)
        
        return book_emb


class user(torch.nn.Module):
    def __init__(self, config):
        super(user, self).__init__()
        self.num_users = config.num_users
        self.num_gender = config.num_gender
        self.num_age = config.num_age
        self.embedding_dim = config.embedding_dim
        


        # self.embedding_users = torch.nn.Linear(
        #     in_features=self.num_users,
        #     out_features=self.embedding_dim,
        #     bias=False
        # )

        self.embedding_users = torch.nn.Embedding(
            num_embeddings=self.num_users,
            embedding_dim=self.embedding_dim
        )

        self.embedding_gender = torch.nn.Embedding(
            num_embeddings=self.num_gender,
            embedding_dim=self.embedding_dim
        )

        self.embedding_age = torch.nn.Embedding(
            num_embeddings=self.num_age,
            embedding_dim=self.embedding_dim
        )

    def forward(self, user_idx, gender_idx, age_idx):
        user_emb = self.embedding_users(user_idx)
        gender_emb = self.embedding_gender(gender_idx)
        age_emb = self.embedding_age(age_idx)
        
        return torch.cat((user_emb, gender_emb, age_emb), 1)
